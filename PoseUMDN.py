import PoseCommon
import PoseTools
import tensorflow as tf
import PoseUNet
import os
import sys
import math
from batch_norm import batch_norm_new as batch_norm
import convNetBase as CNB
import numpy as np

def softmax(x,axis=0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

class PoseUMDN(PoseCommon.PoseCommon):

    def __init__(self, conf, name='pose_umdn',net_type='conv',
                 unet_name = 'pose_unet'):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.dep_nets = PoseUNet.PoseUNet(conf, unet_name)
        self.net_type = net_type
        self.net_name = 'pose_umdn'

    def create_ph(self):
        PoseCommon.PoseCommon.create_ph(self)
        self.dep_nets.create_ph()
        self.ph['x'] = self.dep_nets.ph['x']
        self.ph['locs'] = tf.placeholder(tf.float32,
                           [None, self.conf.n_classes, 2],
                           name='locs')

    def create_fd(self):
        x_shape = [self.conf.batch_size,] + self.ph['x'].get_shape().as_list()[1:]
        l_shape = [self.conf.batch_size,] + self.ph['locs'].get_shape().as_list()[1:]

        self.fd = {self.ph['x']:np.zeros(x_shape),
                   self.ph['locs']:np.zeros(l_shape),
                   self.ph['phase_train']:False,
                   self.ph['learning_rate']:0.
                   }
        self.dep_nets.create_fd()
        self.fd.update(self.dep_nets.fd)

    def create_network(self):
        X = self.dep_nets.create_network()
        if not self.joint:
            X = tf.stop_gradient(X)

        with tf.variable_scope(self.net_name):
            if self.net_type is 'fixed':
                return self.create_network_fixed(X)
            else:
                return self.create_network1(X)

    def create_network1(self, X):

        n_conv = 3
        conv = PoseCommon.conv_relu3
        layers = []
        mdn_prev = None
        n_out = self.conf.n_classes
        k = 2
        locs_offset = 2**len(self.dep_nets.up_layers)
        n_layers_u = len(self.dep_nets.up_layers)

        # MDN downsample.
        for ndx in range(n_layers_u):
            cur_ul = self.dep_nets.up_layers[n_layers_u - ndx - 1]
            cur_dl = self.dep_nets.down_layers[ndx]
            cur_l = tf.concat([cur_ul,cur_dl],axis=3)

            n_filt = cur_l.get_shape().as_list()[3]/2

            if mdn_prev is None:
                X = cur_l
            else:
                X = tf.concat([X, cur_l], axis=3)

            for c_ndx in range(n_conv-1):
                sc_name = 'mdn_{}_{}'.format(ndx,c_ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'])

            # downsample using strides instead of max-pooling
            sc_name = 'mdn_{}_{}'.format(ndx, n_conv)
            with tf.variable_scope(sc_name):
                kernel_shape = [3, 3, n_filt, n_filt]
                weights = tf.get_variable("weights_{}".format(ndx), kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0.))
                cur_conv = tf.nn.conv2d(X, weights, strides=[1, 2, 2, 1], padding='SAME')
                cur_conv = batch_norm(cur_conv, decay=0.99, is_training=self.ph['phase_train'])
                X = tf.nn.relu(cur_conv + biases)


        # few more convolution for the outputs
        n_filt = X.get_shape().as_list()[3]
        with tf.variable_scope('locs'):
            with tf.variable_scope('layer_locs'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv_l = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv_l = batch_norm(conv_l, decay=0.99,
                                  is_training=self.ph['phase_train'])
            mdn_l = tf.nn.relu(conv_l + biases)

            weights_locs = tf.get_variable("weights_locs", [1, 1, n_filt, 2 * k * n_out],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_locs = tf.get_variable("biases_locs", 2 * k * n_out,
                                          initializer=tf.constant_initializer(0))
            o_locs = tf.nn.conv2d(mdn_l, weights_locs,
                                  [1, 1, 1, 1], padding='SAME') + biases_locs

            loc_shape = o_locs.get_shape().as_list()
            n_x = loc_shape[2]
            n_y = loc_shape[1]
            o_locs = tf.reshape(o_locs,[-1, n_y, n_x, k, n_out, 2])
            # when initialized o_locs will be centered around 0 with var 1.
            # with multiplying grid_size/2, o_locs will have variance grid_size/2
            # with adding grid_size/2, o_locs initially will be centered
            # in the center of the grid.
            o_locs *= locs_offset/2
            o_locs += locs_offset/2

            # adding offset of each grid location.
            x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
            x_off = x_off * locs_offset
            y_off = y_off * locs_offset
            x_off = x_off[np.newaxis,:,:,np.newaxis,np.newaxis]
            y_off = y_off[np.newaxis,:,:,np.newaxis,np.newaxis]
            x_locs = o_locs[:,:,:,:,:,0] + x_off
            y_locs = o_locs[:,:,:,:,:,1] + y_off
            o_locs = tf.stack([x_locs, y_locs], axis=5)
            locs = tf.reshape(o_locs,[-1, n_x*n_y*k,n_out,2])

        with tf.variable_scope('scales'):
            with tf.variable_scope('layer_scales'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train'])
            mdn_l = tf.nn.relu(conv + biases)

            weights_scales = tf.get_variable("weights_scales", [1, 1, n_filt, k * n_out],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_scales = tf.get_variable("biases_scales", k * self.conf.n_classes,
                                          initializer=tf.constant_initializer(0))
            o_scales = tf.exp(tf.nn.conv2d(mdn_l, weights_scales,
                                  [1, 1, 1, 1], padding='SAME') + biases_scales)
            # when initialized o_scales will be centered around exp(0) and
            # mostly between [exp(-1), exp(1)] = [0.3 2.7]
            # so adding appropriate offsets to make it lie between the wanted range
            min_sig = self.conf.mdn_min_sigma
            max_sig = self.conf.mdn_max_sigma
            o_scales = (o_scales-1) * (max_sig-min_sig)/2
            o_scales = o_scales + (max_sig-min_sig)/2 + min_sig
            scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
            scales = tf.reshape(scales, [-1, n_x*n_y,k,n_out])
            scales = tf.reshape(scales,[-1, n_x*n_y*k,n_out])

        with tf.variable_scope('logits'):
            with tf.variable_scope('layer_logits'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train'])
            mdn_l = tf.nn.relu(conv + biases)

            weights_logits = tf.get_variable("weights_logits", [1, 1, n_filt, k],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_logits = tf.get_variable("biases_logits", k ,
                                          initializer=tf.constant_initializer(0))
            logits = tf.nn.conv2d(mdn_l, weights_logits,
                                  [1, 1, 1, 1], padding='SAME') + biases_logits
            logits = tf.reshape(logits, [-1, n_x * n_y, k])
            logits = tf.reshape(logits, [-1, n_x * n_y * k])

        return [locs,scales,logits]

    def create_network_fixed(self, X):
        # Downsample to a size between 4x4 and 2x2

        n_conv = 3
        conv = PoseCommon.conv_relu3
        layers = []
        mdn_prev = None
        n_out = self.conf.n_classes
        k = 10 # this is the number of final gaussians.
        n_layers_u = len(self.dep_nets.up_layers)
        min_im_sz = min(self.conf.imsz)
        extra_layers = math.floor(np.log2(min_im_sz)) - n_layers_u - 2
        extra_layers = int(extra_layers)
        locs_offset = 2**(len(self.dep_nets.up_layers) + extra_layers)

        # MDN downsample.
        for ndx in range(n_layers_u):
            cur_ul = self.dep_nets.up_layers[n_layers_u - ndx - 1]
            cur_dl = self.dep_nets.down_layers[ndx]
            cur_l = tf.concat([cur_ul,cur_dl],axis=3)

            n_filt = cur_l.get_shape().as_list()[3]/2

            if mdn_prev is None:
                X = cur_l
            else:
                X = tf.concat([X, cur_l], axis=3)

            for c_ndx in range(n_conv-1):
                sc_name = 'mdn_{}_{}'.format(ndx,c_ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'])

            # downsample using strides instead of max-pooling
            sc_name = 'mdn_{}_{}'.format(ndx, n_conv)
            with tf.variable_scope(sc_name):
                kernel_shape = [3, 3, n_filt, n_filt]
                weights = tf.get_variable("weights_{}".format(ndx), kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0.))
                cur_conv = tf.nn.conv2d(X, weights, strides=[1, 2, 2, 1], padding='SAME')
                cur_conv = batch_norm(cur_conv, decay=0.99, is_training=self.ph['phase_train'])
                X = tf.nn.relu(cur_conv + biases)

        # few more convolution for the outputs
        n_filt = X.get_shape().as_list()[3]
        for ndx in range(extra_layers):
            for c_ndx in range(n_conv-1):
                sc_name = 'mdn_extra_{}_{}'.format(ndx,c_ndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'])

            with tf.variable_scope('mdn_extra_{}'.format(ndx)):
                kernel_shape = [3, 3, n_filt, n_filt]
                weights = tf.get_variable("weights_{}".format(ndx), kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0.))
                cur_conv = tf.nn.conv2d(X, weights, strides=[1, 2, 2, 1], padding='SAME')
                cur_conv = batch_norm(cur_conv, decay=0.99, is_training=self.ph['phase_train'])
                X = tf.nn.relu(cur_conv + biases)

        X_sz = min(X.get_shape().as_list()[1:3])
        assert (X_sz >= 4) and (X_sz<8), 'The net has been reduced too much or not too much'

        # few more convolution for the outputs
        n_filt = X.get_shape().as_list()[3]
        with tf.variable_scope('locs'):
            with tf.variable_scope('layer_locs'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv_l = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv_l = batch_norm(conv_l, decay=0.99,
                                  is_training=self.ph['phase_train'])
            mdn_l = tf.nn.relu(conv_l + biases)

            weights_locs = tf.get_variable("weights_locs", [1, 1, n_filt, 2 * k * n_out],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_locs = tf.get_variable("biases_locs", 2 * k * n_out,
                                          initializer=tf.constant_initializer(0))
            o_locs = tf.nn.conv2d(mdn_l, weights_locs,
                                  [1, 1, 1, 1], padding='SAME') + biases_locs

            loc_shape = o_locs.get_shape().as_list()
            n_x = loc_shape[2]
            n_y = loc_shape[1]
            o_locs = tf.reshape(o_locs,[-1, n_y, n_x, k, n_out, 2])
            # when initialized o_locs will be centered around 0 with var 1.
            # with multiplying grid_size/2, o_locs will have variance grid_size/2
            # with adding grid_size/2, o_locs initially will be centered
            # in the center of the grid.
            o_locs *= locs_offset/2
            o_locs += locs_offset/2

            # adding offset of each grid location.
            x_off, y_off = np.meshgrid(np.arange(loc_shape[2]), np.arange(loc_shape[1]))
            x_off = x_off * locs_offset
            y_off = y_off * locs_offset
            x_off = x_off[np.newaxis,:,:,np.newaxis,np.newaxis]
            y_off = y_off[np.newaxis,:,:,np.newaxis,np.newaxis]
            x_locs = o_locs[:,:,:,:,:,0] + x_off
            y_locs = o_locs[:,:,:,:,:,1] + y_off
            o_locs = tf.stack([x_locs, y_locs], axis=5)
            locs = tf.reshape(o_locs,[-1, n_x*n_y*k,n_out,2])

        with tf.variable_scope('scales'):
            with tf.variable_scope('layer_scales'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train'])
            mdn_l = tf.nn.relu(conv + biases)

            weights_scales = tf.get_variable("weights_scales", [1, 1, n_filt, k * n_out],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_scales = tf.get_variable("biases_scales", k * self.conf.n_classes,
                                          initializer=tf.constant_initializer(0))
            o_scales = tf.exp(tf.nn.conv2d(mdn_l, weights_scales,
                                  [1, 1, 1, 1], padding='SAME') + biases_scales)
            # when initialized o_scales will be centered around exp(0) and
            # mostly between [exp(-1), exp(1)] = [0.3 2.7]
            # so adding appropriate offsets to make it lie between the wanted range
            min_sig = self.conf.mdn_min_sigma
            max_sig = self.conf.mdn_max_sigma
            o_scales = (o_scales-1) * (max_sig-min_sig)/2
            o_scales = o_scales + (max_sig-min_sig)/2 + min_sig
            scales = tf.minimum(max_sig, tf.maximum(min_sig, o_scales))
            scales = tf.reshape(scales, [-1, n_x*n_y,k,n_out])
            scales = tf.reshape(scales,[-1, n_x*n_y*k,n_out])

        with tf.variable_scope('logits'):
            with tf.variable_scope('layer_logits'):
                kernel_shape = [1, 1, n_filt, n_filt]
                weights = tf.get_variable("weights", kernel_shape,
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable("biases", kernel_shape[-1],
                                         initializer=tf.constant_initializer(0))
                conv = tf.nn.conv2d(X, weights,
                                    strides=[1, 1, 1, 1], padding='SAME')
                conv = batch_norm(conv, decay=0.99,
                                  is_training=self.ph['phase_train'])
            mdn_l = tf.nn.relu(conv + biases)

            weights_logits = tf.get_variable("weights_logits", [1, 1, n_filt, k],
                                           initializer=tf.contrib.layers.xavier_initializer())
            biases_logits = tf.get_variable("biases_logits", k ,
                                          initializer=tf.constant_initializer(0))
            logits = tf.nn.conv2d(mdn_l, weights_logits,
                                  [1, 1, 1, 1], padding='SAME') + biases_logits
            logits = tf.reshape(logits, [-1, n_x * n_y, k])
            logits = tf.reshape(logits, [-1, n_x * n_y * k])

        return [locs,scales,logits]

    def fd_train(self):
        self.fd[self.ph['phase_train']] = True
        self.dep_nets.fd[self.ph['phase_train']] = self.joint

    def fd_val(self):
        self.fd[self.ph['phase_train']] = False
        self.dep_nets.fd[self.ph['phase_train']] = False

    def update_fd(self, db_type, sess, distort):
        self.read_images(db_type, distort, sess, distort)
        rescale = self.conf.unet_rescale
        xs = PoseTools.scale_images(self.xs, rescale, self.conf)
        self.fd[self.ph['x']] = PoseTools.normalize_mean(xs, self.conf)
        self.locs[np.isnan(self.locs)] = -500000.
        self.fd[self.ph['locs']] = self.locs/rescale

    def my_loss(self, X, y):
        mdn_locs, mdn_scales, mdn_logits = X
        cur_comp = []
        ll = tf.nn.softmax(mdn_logits, dim=1)
        # All gaussians in the mixture have some weight so that all the mixtures try to predict correctly.
        logit_eps = self.conf.mdn_logit_eps_training
        ll = tf.cond(self.ph['phase_train'], lambda: ll + logit_eps, lambda: tf.identity(ll))
        ll = ll / tf.reduce_sum(ll, axis=1, keep_dims=True)
        for cls in range(self.conf.n_classes):
            cur_scales = mdn_scales[:, :, cls]
            pp = y[:, cls:cls + 1, :]
            kk = tf.reduce_sum(tf.square(pp - mdn_locs[:, :, cls, :]), axis=2)
            cur_comp.append(tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2)))

        cur_comp = tf.stack(cur_comp, 1)
        cur_loss = tf.reduce_prod(cur_comp, axis=1)
        # product because we are looking at joint distribution of all the points.
        pp = (cur_loss * ll) + 1e-30
        loss = -tf.log(tf.reduce_sum(pp, axis=1))
        return tf.reduce_sum(loss)

    def compute_dist(self, preds, locs):
        locs = locs.copy()
        if locs.ndim == 3:
            locs = locs[:,np.newaxis,:,:]
        val_means, val_std, val_wts = preds
        val_dist = np.zeros(locs.shape[:-1])
        pred_dist = np.zeros(val_means.shape[:-1])
        pred_dist[:] = np.nan
        for ndx in range(val_means.shape[0]):
            sel_ex = val_wts[ndx, :] > 0
            jj = val_means[ndx, ...][np.newaxis, sel_ex, :, :] - \
                 locs[ndx, ...][:, np.newaxis, :, :]
            # jj has distance between all labels and
            # all predictions with wts > 0.
            dd1 = np.sqrt(np.sum(jj ** 2, axis=-1)).min(axis=1)
            # instead of min it should ideally be matching.
            # but for now this is ok.
            dd2 = np.sqrt(np.sum(jj ** 2, axis=-1)).min(axis=0)
            # dd1 -- to every label is there a close by prediction.
            # dd2 -- to every prediction is there a close by labels.
            # or dd1 is fn and dd2 is fp.
            val_dist[ndx,:, :] = dd1
            pred_dist[ndx,sel_ex, :] = dd2
        val_dist[locs[..., 0] < -5000] = np.nan
        pred_mean = np.nanmean(pred_dist)
        label_mean = np.nanmean(val_dist)
        return (pred_mean+label_mean)/2

    def compute_train_data(self, sess, db_type):
        self.setup_train(sess) if db_type is self.DBType.Train \
            else self.setup_val(sess)
        p_m, p_s, p_w = self.pred
        cur_loss, pred_means, pred_std, pred_weights = sess.run(
            [self.cost, p_m, p_s, tf.nn.softmax(p_w,dim=1)], self.fd)

        cur_dist = self.compute_dist(
            [pred_means, pred_std, pred_weights], self.locs)
        return cur_loss, cur_dist

    def train_umdn(self, restore, train_type=0,joint=False,
                   net_type='conv'):

        self.joint = joint
        self.net_type = net_type
        if joint:
            training_iters = 40000/self.conf.batch_size*8
        else:
            training_iters = 20000/self.conf.batch_size*8

        super(self.__class__, self).train(
            restore=restore,
            train_type=train_type,
            create_network=self.create_network,
            training_iters=training_iters,
            loss=self.my_loss,
            pred_in_key='locs',
            learning_rate=0.0001,
            td_fields=('loss','dist'))

    def classify_val(self):
        val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        num_val = 0
        for record in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        self.init_train(train_type=0)
        self.pred = self.create_network()
        saver = self.create_saver()
        p_m, p_s, p_w = self.pred
        conf = self.conf
        osz = self.conf.imsz
        self.joint = True

        with tf.Session() as sess:
            start_at = self.init_and_restore(sess, True, ['loss', 'dist'])

            val_dist = []
            val_ims = []
            val_preds = []
            val_predlocs = []
            val_locs = []
            val_means = []
            val_std = []
            val_wts = []
            for step in range(num_val/self.conf.batch_size):
                self.setup_val(sess)
                pred_means, pred_std, pred_weights = sess.run(
                    [p_m, p_s, p_w], self.fd)
                val_means.append(pred_means)
                val_std.append(pred_std)
                val_wts.append(pred_weights)
                pred_weights = softmax(pred_weights,axis=1)
                mdn_pred_out = np.zeros([self.conf.batch_size, osz[0], osz[1], conf.n_classes])
                for sel in range(conf.batch_size):
                    for cls in range(conf.n_classes):
                        for ndx in range(pred_means.shape[1]):
                            if pred_weights[sel, ndx] < (0.02/self.conf.max_n_animals):
                                continue
                            cur_locs = pred_means[sel:sel + 1, ndx:ndx + 1, cls, :].astype('int')
                            # cur_scale = pred_std[sel, ndx, cls, :].mean().astype('int')
                            cur_scale = pred_std[sel, ndx, cls].astype('int')
                            curl = (PoseTools.create_label_images(cur_locs, osz, 1, cur_scale) + 1) / 2
                            mdn_pred_out[sel, :, :, cls] += pred_weights[sel, ndx] * curl[0, ..., 0]

                if self.locs.ndim == 3:
                    cur_predlocs = PoseTools.get_pred_locs(mdn_pred_out)
                    cur_dist = np.sqrt(np.sum(
                        (cur_predlocs - self.locs) ** 2, 2))
                else:
                    cur_predlocs = PoseTools.get_pred_locs_multi(
                        mdn_pred_out,self.conf.max_n_animals,
                        self.conf.label_blur_rad * 7)
                    curl = self.locs.copy()
                    jj = cur_predlocs[:,:,np.newaxis,:,:] - curl[:,np.newaxis,...]
                    cur_dist = np.sqrt(np.sum(jj**2,axis=-1)).min(axis=1)
                val_dist.append(cur_dist)
                val_ims.append(self.xs)
                val_locs.append(self.locs)
                val_preds.append(mdn_pred_out)
                val_predlocs.append(cur_predlocs)
            self.close_cursors()

        def val_reshape(in_a):
            in_a = np.array(in_a)
            return in_a.reshape( (-1,) + in_a.shape[2:])
        val_dist = val_reshape(val_dist)
        val_ims = val_reshape(val_ims)
        val_preds = val_reshape(val_preds)
        val_predlocs = val_reshape(val_predlocs)
        val_locs = val_reshape(val_locs)
        val_means = val_reshape(val_means)
        val_std = val_reshape(val_std)
        val_wts = val_reshape(val_wts)

        return val_dist, val_ims, val_preds, val_predlocs, val_locs,\
               [val_means,val_std,val_wts]


class PoseUMDNMulti(PoseUMDN, PoseCommon.PoseCommonMulti):

    def __init__(self, conf, name='pose_umdn_multi',net_type='conv',
                 unet_name = 'pose_unet_multi'):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.dep_nets = PoseUNet.PoseUNetMulti(conf, unet_name)
        self.net_type = net_type

    def create_cursors(self, sess):
        PoseCommon.PoseCommonMulti.create_cursors(self, sess)

    def create_ph(self):
        PoseCommon.PoseCommon.create_ph(self)
        self.dep_nets.create_ph()
        self.ph['x'] = self.dep_nets.ph['x']
        self.ph['locs'] = tf.placeholder(tf.float32,
                           [None, self.conf.max_n_animals, self.conf.n_classes, 2],
                           name='locs')

    def loss_multi(self, X, y):
        mdn_locs, mdn_scales, mdn_logits = X
        cur_comp = []

        # softmax of logits
        ll = tf.nn.softmax(mdn_logits, dim=1)
        # All gaussians in the mixture have some weight so that all the mixtures try to predict correctly.
        logit_eps = self.conf.mdn_logit_eps_training
        ll = tf.cond(self.ph['phase_train'], lambda: ll + logit_eps, lambda: tf.identity(ll))
        ll = ll / tf.reduce_sum(ll, axis=1, keep_dims=True)

        # find distance of each predicted gaussian to all the labels.
        # soft assign each prediction to the labels.
        # compute loss based on this soft assignment.
        # also the total weight that gets assigned to each label should
        # be roughly equal. Add loss for that too.

        labels_valid = tf.cast(y[:, :, 0, 0] > -10000.,tf.float32)
        n_labels = tf.reduce_sum(labels_valid,axis=1)
        # labels that don't exist have high negative values
        # distance of predicted gaussians from such labels should be high
        # and they shouldn't create problems with soft_assign.
        for cls in range(self.conf.n_classes):
            cur_scales = mdn_scales[:, :, cls:cls+1]
            pp = tf.expand_dims(y[:, :, cls, :],axis=1)
            kk = tf.reduce_sum(tf.square(pp - mdn_locs[:, :, cls:cls+1, :]), axis=3)
            # kk now has the distance between all the labels and all the predictions.
            dd = tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2))
            cur_comp.append(dd)

        cur_comp = tf.stack(cur_comp, 1)
        tot_dist = tf.reduce_prod(cur_comp, axis=1)
        # soft assignment to each label.
        soft_assign = tf.expand_dims(ll,axis=2) * tf.nn.softmax(tot_dist,dim=2)
        ind_loss = tot_dist * soft_assign

        # how much weight does each label have.
        # all the valid labels should get equal weights
        # without this, the net might predict only label.
        lbl_weights = tf.reduce_sum(soft_assign,axis=1)
        lbl_weights_loss = (1-lbl_weights*tf.expand_dims(n_labels,axis=1))**2

        lbl_weights_loss = lbl_weights_loss*labels_valid

        # for each label, the predictions are treated as mixture.
        int_loss = -tf.log(tf.reduce_sum(ind_loss, axis=1)+ 1e-30)
        loss = int_loss * lbl_weights_loss
        return tf.reduce_sum(loss)

    def train_umdn(self, restore, train_type=0,joint=False,
                   net_type='conv'):

        self.joint = joint
        self.net_type = net_type
        if joint:
            training_iters = 40000/self.conf.batch_size*8
        else:
            training_iters = 20000/self.conf.batch_size*8

        super(self.__class__, self).train(
            restore=restore,
            train_type=train_type,
            create_network=self.create_network,
            training_iters=training_iters,
            loss=self.loss_multi,
            pred_in_key='locs',
            learning_rate=0.0001,
            td_fields=('loss','dist'))

