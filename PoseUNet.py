import PoseCommon
import PoseTools
import tensorflow as tf
import os
import sys
import math
from tensorflow.contrib.layers import batch_norm
import convNetBase as CNB
import numpy as np
import movies
from PoseTools import scale_images
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tempfile
from matplotlib import cm
import movies
import multiResData
from scipy import io as sio
# for tf_unet
#from tf_unet_layers import (weight_variable, weight_variable_devonc, bias_variable,
#                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
#                            cross_entropy)
from collections import OrderedDict

class PoseUNet(PoseCommon.PoseCommon):

    def __init__(self, conf, name='pose_unet'):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.down_layers = [] # layers created while down sampling
        self.up_layers = [] # layers created while up sampling
        self.edge_ignore = 10
        self.net_name = 'pose_unet'
        self.keep_prob = 0.7

    def create_ph(self):
        PoseCommon.PoseCommon.create_ph(self)
        imsz = self.conf.imsz
        rescale = self.conf.unet_rescale
        self.ph['x'] = tf.placeholder(tf.float32,
                           [None,imsz[0]/rescale,imsz[1]/rescale, self.conf.imgDim],
                           name='x')
        self.ph['y'] = tf.placeholder(tf.float32,
                           [None,imsz[0]/rescale,imsz[1]/rescale, self.conf.n_classes],
                           name='y')
        self.ph['keep_prob'] = tf.placeholder(tf.float32)

    def create_fd(self):
        x_shape = [self.conf.batch_size,] + self.ph['x'].get_shape().as_list()[1:]
        y_shape = [self.conf.batch_size,] + self.ph['y'].get_shape().as_list()[1:]
        self.fd = {self.ph['x']:np.zeros(x_shape),
                   self.ph['y']:np.zeros(y_shape),
                   self.ph['phase_train']:False,
                   self.ph['learning_rate']:0.
                   }

    def create_network(self):
        with tf.variable_scope(self.net_name):
            return self.create_network1()

    def compute_dist(self, preds, locs):
        tt1 = PoseTools.get_pred_locs(preds,self.edge_ignore) - \
              locs/self.conf.unet_rescale
        tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
        return np.nanmean(tt1)

    def create_network1(self):
        m_sz = min(self.conf.imsz)/self.conf.unet_rescale
        max_layers = int(math.ceil(math.log(m_sz,2)))-1
        sel_sz = self.conf.sel_sz
        n_layers = int(math.ceil(math.log(sel_sz,2)))+2
        # max_layers = int(math.floor(math.log(m_sz)))
        # sel_sz = self.conf.sel_sz
        # n_layers = int(math.ceil(math.log(sel_sz)))+2
        n_layers = min(max_layers,n_layers) - 2

        n_conv = 2 #3
        conv = PoseCommon.conv_relu3
        # conv = PoseCommon.conv_relu3_noscaling
        layers = []
        up_layers = []
        layers_sz = []
        X = self.ph['x']
        n_out = self.conf.n_classes
        debug_layers = []

        # downsample
        for ndx in range(n_layers):
            if ndx is 0:
                n_filt = 64 #32
            elif ndx is 1:
                n_filt = 128 #64
            elif ndx is 2:
                n_filt = 256
            else:
                n_filt = 512 #128

            for cndx in range(n_conv):
                sc_name = 'layerdown_{}_{}'.format(ndx,cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
                debug_layers.append(X)
            layers.append(X)
            layers_sz.append(X.get_shape().as_list()[1:3])
            # X = tf.nn.max_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],
            #                    padding='SAME')
            X = tf.nn.avg_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME')

        self.down_layers = layers
        self.debug_layers = debug_layers
        # few more convolution for the final layers
        for cndx in range(n_conv):
            sc_name = 'layer_{}_{}'.format(n_layers,cndx)
            with tf.variable_scope(sc_name):
                X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])

        # upsample
        for ndx in reversed(range(n_layers)):
            # if ndx is 0:
            #     n_filt = 64
            # elif ndx is 1:
            #     n_filt = 64
            # # elif ndx is 2:
            # #     n_filt = 256
            # else:
            #     n_filt = 128
            if ndx is 0:
                n_filt = 64  # 32
            elif ndx is 1:
                n_filt = 128  # 64
            elif ndx is 2:
                n_filt = 256
            else:
                n_filt = 512  # 128
            X = CNB.upscale('u_'.format(ndx), X, layers_sz[ndx])
            X = tf.concat([X,layers[ndx]], axis=3)
            for cndx in range(n_conv):
                sc_name = 'layerup_{}_{}'.format(ndx, cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
            up_layers.append(X)
        self.up_layers = up_layers

        # final conv
        weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("out_biases", n_out,
                                 initializer=tf.constant_initializer(0.))
        conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
        X = conv + biases
        return X

    # def create_network_tf_unet(self):
    #     # implementation of unet from https://github.com/jakeret/tf_unet/blob/master/tf_unet/unet.py
    #     x = self.ph['x']
    #     nx = tf.shape(x)[1]
    #     ny = tf.shape(x)[2]
    #     # x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
    #     x_image = x
    #     in_node = x_image
    #     batch_size = tf.shape(x_image)[0]
    #
    #     weights = []
    #     biases = []
    #     convs = []
    #     pools = OrderedDict()
    #     deconv = OrderedDict()
    #     dw_h_convs = OrderedDict()
    #     up_h_convs = OrderedDict()
    #
    #     layers = 5
    #     features_root = 16
    #     filter_size = 3
    #     pool_size = 2
    #     channels = self.conf.imgDim
    #     keep_prob = 1.
    #     n_class = self.conf.n_classes
    #
    #     in_size = 1000
    #     size = in_size
    #     # down layers
    #     for layer in range(0, layers):
    #         features = 2 ** layer * features_root
    #         stddev = np.sqrt(2 / (filter_size ** 2 * features))
    #         if layer == 0:
    #             w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
    #         else:
    #             w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev)
    #
    #         w2 = weight_variable([filter_size, filter_size, features, features], stddev)
    #         b1 = bias_variable([features])
    #         b2 = bias_variable([features])
    #
    #         conv1 = conv2d(in_node, w1, keep_prob)
    #         tmp_h_conv = tf.nn.relu(conv1 + b1)
    #         conv2 = conv2d(tmp_h_conv, w2, keep_prob)
    #         dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
    #
    #         weights.append((w1, w2))
    #         biases.append((b1, b2))
    #         convs.append((conv1, conv2))
    #
    #         size -= 4
    #         if layer < layers - 1:
    #             pools[layer] = max_pool(dw_h_convs[layer], pool_size)
    #             in_node = pools[layer]
    #             size /= 2
    #
    #     in_node = dw_h_convs[layers - 1]
    #
    #     # up layers
    #     for layer in range(layers - 2, -1, -1):
    #         features = 2 ** (layer + 1) * features_root
    #         stddev = np.sqrt(2 / (filter_size ** 2 * features))
    #
    #         wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev)
    #         bd = bias_variable([features // 2])
    #         h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
    #         h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
    #         deconv[layer] = h_deconv_concat
    #
    #         w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev)
    #         w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev)
    #         b1 = bias_variable([features // 2])
    #         b2 = bias_variable([features // 2])
    #
    #         conv1 = conv2d(h_deconv_concat, w1, keep_prob)
    #         h_conv = tf.nn.relu(conv1 + b1)
    #         conv2 = conv2d(h_conv, w2, keep_prob)
    #         in_node = tf.nn.relu(conv2 + b2)
    #         up_h_convs[layer] = in_node
    #
    #         weights.append((w1, w2))
    #         biases.append((b1, b2))
    #         convs.append((conv1, conv2))
    #
    #         size *= 2
    #         size -= 4
    #
    #     # Output Map
    #     weight = weight_variable([1, 1, features_root, n_class], stddev)
    #     bias = bias_variable([n_class])
    #     conv = conv2d(in_node, weight, tf.constant(1.0))
    #     output_map = tf.nn.relu(conv + bias)
    #     up_h_convs["out"] = output_map
    #
    #     variables = []
    #     for w1, w2 in weights:
    #         variables.append(w1)
    #         variables.append(w2)
    #
    #     for b1, b2 in biases:
    #         variables.append(b1)
    #         variables.append(b2)
    #
    #     return output_map #, variables, int(in_size - size)

    def fd_train(self):
        self.fd[self.ph['phase_train']] = True
        self.fd[self.ph['keep_prob']] = self.keep_prob

    def fd_val(self):
        self.fd[self.ph['phase_train']] = False
        self.fd[self.ph['keep_prob']] = 1.

    def update_fd(self, db_type, sess, distort):
        self.read_images(db_type, distort, sess, distort)
        rescale = self.conf.unet_rescale
        xs = scale_images(self.xs, rescale, self.conf)
        self.fd[self.ph['x']] = PoseTools.normalize_mean(xs, self.conf)
        imsz = [self.conf.imsz[0]/rescale, self.conf.imsz[1]/rescale,]
        label_ims = PoseTools.create_label_images(
            self.locs/rescale, imsz, 1, self.conf.label_blur_rad)
        self.fd[self.ph['y']] = label_ims

    def init_net(self, train_type=0, restore=True):
        self.init_train(train_type=train_type)
        self.pred = self.create_network()
        saver = self.create_saver()
        self.joint = True

        sess = tf.InteractiveSession()
        start_at = self.init_and_restore(sess, restore, ['loss', 'dist'])
        return sess

    def train_unet(self, restore, train_type=0):

        def loss(pred_in, pred_out):
            return tf.nn.l2_loss(pred_in-pred_out)

        PoseCommon.PoseCommon.train(self,
            restore=restore,
            train_type=train_type,
            create_network=self.create_network,
            training_iters=12000/self.conf.batch_size*8,
            loss=loss,
            pred_in_key='y',
            learning_rate=0.0001,
            td_fields=('loss','dist'))

    def classify_val(self, train_type=0, at_step=-1):
        if train_type is 0:
            val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        else:
            val_file = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename + '.tfrecords')
        num_val = 0
        for _ in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        self.init_train(train_type)
        self.pred = self.create_network()
        self.create_saver()

        with tf.Session() as sess:
            self.init_and_restore(sess, True, ['loss', 'dist'], at_step)
            val_dist = []
            val_ims = []
            val_preds = []
            val_predlocs = []
            val_locs = []
            for step in range(num_val/self.conf.batch_size):
                self.setup_val(sess)
                cur_pred = sess.run(self.pred, self.fd)
                cur_predlocs = PoseTools.get_pred_locs(
                    cur_pred, self.edge_ignore)
                cur_dist = np.sqrt(np.sum(
                    (cur_predlocs-self.locs/self.conf.unet_rescale) ** 2, 2))
                val_dist.append(cur_dist)
                val_ims.append(self.xs)
                val_locs.append(self.locs)
                val_preds.append(cur_pred)
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
        tf.reset_default_graph()
        return val_dist, val_ims, val_preds, val_predlocs, val_locs/self.conf.unet_rescale

    def classify_movie(self, movie_name, sess, max_frames=-1, start_at=0, flipud=False):
        # maxframes if specificied reads that many frames
        # start at specifies where to start reading.
        conf = self.conf

        cap = movies.Movie(movie_name)
        n_frames = int(cap.get_n_frames())

        # figure out how many frames to read
        if max_frames > 0:
            if max_frames + start_at > n_frames:
                n_frames = n_frames - start_at
            else:
                n_frames = max_frames
        else:
            n_frames = n_frames - start_at


        # pre allocate results
        bsize = conf.batch_size
        n_batches = int(math.ceil(float(n_frames)/ bsize))
        pred_locs = np.zeros([n_frames, conf.n_classes, 2])
        pred_max_scores = np.zeros([n_frames, conf.n_classes])
        pred_scores = np.zeros([n_frames,] + self.pred.get_shape().as_list()[1:])
        all_f = np.zeros((bsize,) + conf.imsz + (1,))

        for curl in range(n_batches):
            ndx_start = curl * bsize
            ndx_end = min(n_frames, (curl + 1) * bsize)
            ppe = min(ndx_end - ndx_start, bsize)
            for ii in range(ppe):
                fnum = ndx_start + ii + start_at
                frame_in = cap.get_frame(fnum)
                if len(frame_in) == 2:
                    frame_in = frame_in[0]
                    if frame_in.ndim == 2:
                        frame_in = frame_in[:, :, np.newaxis]
                frame_in = PoseTools.crop_images(frame_in, conf)
                if flipud:
                    frame_in = np.flipud(frame_in)
                all_f[ii, ...] = frame_in[..., 0:conf.imgDim]

            xs = PoseTools.adjust_contrast(all_f, conf)
            xs = PoseTools.scale_images(xs, conf.unet_rescale, conf)
            xs = PoseTools.normalize_mean(xs, self.conf)

            self.fd[self.ph['x']] = xs
            self.fd[self.ph['phase_train']] = False
            pred = sess.run(self.pred, self.fd)

            base_locs = PoseTools.get_base_pred_locs(pred, conf)
            base_locs = base_locs*conf.unet_rescale
            pred_locs[ndx_start:ndx_end, :, :] = base_locs[:ppe, :, :]
            pred_max_scores[ndx_start:ndx_end, :] = pred[:ppe, :, :, :].max(axis=(1,2))
            pred_scores[ndx_start:ndx_end, :, :, :] = pred[:ppe, :, :, :]
            sys.stdout.write('.')
            if curl % 20 == 19:
                sys.stdout.write('\n')

        cap.close()
        return pred_locs, pred_scores, pred_max_scores

    def classify_movie_trx(self, movie_name, trx, sess, max_frames=-1, start_at=0,flipud=False, return_ims=False):
        # maxframes if specificied reads that many frames
        # start at specifies where to start reading.

        conf = self.conf
        cap = movies.Movie(movie_name)
        n_frames = int(cap.get_n_frames())
        T = sio.loadmat(trx)['trx'][0]
        n_trx = len(T)

        end_frames = np.array([x['endframe'][0,0] for x in T])
        first_frames = np.array([x['firstframe'][0,0] for x in T]) - 1 # for converting from 1 indexing to 0 indexing
        if max_frames < 0:
            max_frames = end_frames.max()
        if start_at > max_frames:
            return None
        max_n_frames = max_frames - start_at
        pred_locs = np.zeros([max_n_frames, n_trx, conf.n_classes, 2])
        pred_locs[:] = np.nan

        if return_ims:
            ims = np.zeros([max_n_frames, n_trx, conf.imsz[0]/conf.unet_rescale, conf.imsz[1]/conf.unet_rescale,conf.imgDim])

        bsize = conf.batch_size

        for trx_ndx in range(n_trx):
            cur_trx = T[trx_ndx]
            # pre allocate results
            if first_frames[trx_ndx] > start_at:
                cur_start = first_frames[trx_ndx]
            else:
                cur_start = start_at

            if end_frames[trx_ndx] < max_frames:
                cur_end = end_frames[trx_ndx]
            else:
                cur_end = max_frames

            n_frames = cur_end - cur_start
            n_batches = int(math.ceil(float(n_frames)/ bsize))
            all_f = np.zeros((bsize,) + conf.imsz + (conf.imgDim,))

            for curl in range(n_batches):
                ndx_start = curl * bsize + cur_start - start_at
                ndx_end = min(n_frames, (curl + 1) * bsize) + cur_start - start_at
                ppe = min(ndx_end - ndx_start, bsize)
                for ii in range(ppe):
                    fnum = ndx_start + ii + cur_start
                    frame_in = cap.get_frame(fnum)
                    if len(frame_in) == 2:
                        frame_in = frame_in[0]
                        if frame_in.ndim == 2:
                            frame_in = frame_in[:,:,np.newaxis]
                    if flipud:
                        frame_in = np.flipud(frame_in)

                    trx_fnum = fnum - first_frames[trx_ndx]
                    x = int(round(cur_trx['x'][0,trx_fnum]))-1
                    y = int(round(cur_trx['y'][0,trx_fnum]))-1
                    # -1 for 1-indexing in matlab and 0-indexing in python
                    theta = cur_trx['theta'][0,trx_fnum]
                    assert conf.imsz[0] == conf.imsz[1]

                    frame_in, _ = multiResData.get_patch_trx(frame_in,x,y,theta,conf.imsz[0], np.zeros([2,2]))
                    frame_in = frame_in[:, :, 0:conf.imgDim]
                    all_f[ii, ...] = frame_in[..., 0:conf.imgDim]

                xs = PoseTools.adjust_contrast(all_f, conf)
                xs = PoseTools.scale_images(xs, conf.unet_rescale, conf)
                xs = PoseTools.normalize_mean(xs, self.conf)
                self.fd[self.ph['x']] = xs
                self.fd[self.ph['phase_train']] = False
                pred = sess.run(self.pred, self.fd)

                base_locs = PoseTools.get_base_pred_locs(pred, conf)
                base_locs = base_locs * conf.unet_rescale

                if return_ims:
                    ims[ndx_start:ndx_end, trx_ndx,:,:,:] = xs[:ppe,...]

                pred_locs[ndx_start:ndx_end, trx_ndx, :, :] = base_locs[:ppe,...]
                sys.stdout.write('.')
                if curl % 20 == 19:
                    sys.stdout.write('\n')

            sys.stdout.write('\n')
        cap.close()
        tf.reset_default_graph()
        if return_ims:
            return pred_locs, ims
        else:
            return pred_locs

    def create_pred_movie(self, movie_name, out_movie, max_frames=-1, flipud=False, trace=True):
        conf = self.conf
        sess = self.init_net(0,True)
        predLocs, pred_scores, pred_max_scores = self.classify_movie(movie_name,sess,max_frames=max_frames,flipud=flipud)
        tdir = tempfile.mkdtemp()

        cap = movies.Movie(movie_name)
        nframes = int(cap.get_n_frames())
        if max_frames > 0:
            nframes = max_frames

        fig = mpl.figure.Figure(figsize=(9, 4))
        canvas = FigureCanvasAgg(fig)
        sc = self.conf.unet_rescale

        color = cm.hsv(np.linspace(0, 1 - 1./conf.n_classes, conf.n_classes))
        trace_len = 30
        for curl in range(nframes):
            frame_in = cap.get_frame(curl)
            if len(frame_in) == 2:
                frame_in = frame_in[0]
                if frame_in.ndim == 2:
                    frame_in = frame_in[:,:, np.newaxis]
            frame_in = PoseTools.crop_images(frame_in, conf)

            if flipud:
                frame_in = np.flipud(frame_in)
            fig.clf()
            ax1 = fig.add_subplot(1, 1, 1)
            if frame_in.shape[2] == 1:
                ax1.imshow(frame_in[:,:,0], cmap=cm.gray)
            else:
                ax1.imshow(frame_in)
            xlim = ax1.get_xlim()
            ylim = ax1.get_ylim()

            ax1.scatter(predLocs[curl, :, 0],
                        predLocs[curl, :, 1],
                c=color*0.9, linewidths=0,
                edgecolors='face',marker='+',s=45)
            if trace:
                for ndx in range(conf.n_classes):
                    curc = color[ndx,:].copy()
                    curc[3] = 0.5
                    e = np.maximum(0,curl-trace_len)
                    ax1.plot(predLocs[e:curl,ndx,0],
                             predLocs[e:curl, ndx, 1],
                             c = curc,lw=0.8)
            ax1.axis('off')
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            fname = "test_{:06d}.png".format(curl)

            # to printout without X.
            # From: http://www.dalkescientific.com/writings/diary/archive/2005/04/23/matplotlib_without_gui.html
            # The size * the dpi gives the final image size
            #   a4"x4" image * 80 dpi ==> 320x320 pixel image
            canvas.print_figure(os.path.join(tdir, fname), dpi=160)

            # below is the easy way.
        #         plt.savefig(os.path.join(tdir,fname))

        tfilestr = os.path.join(tdir, 'test_*.png')
        mencoder_cmd = "mencoder mf://" + tfilestr + " -frames " + "{:d}".format(
            nframes) + " -mf type=png:fps=15 -o " + out_movie + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
        os.system(mencoder_cmd)
        cap.close()
        tf.reset_default_graph()


class PoseUNetMulti(PoseUNet, PoseCommon.PoseCommonMulti):

    def __init__(self, conf, name='pose_unet_multi'):
        PoseUNet.__init__(self, conf, name)

    def update_fd(self, db_type, sess, distort):
        self.read_images(db_type, distort, sess, distort)
        self.fd[self.ph['x']] = self.xs
        n_classes = self.locs.shape[2]
        sz0 = self.conf.imsz[0]
        sz1 = self.conf.imsz[1]
        label_ims = np.zeros([self.conf.batch_size, sz0, sz1, n_classes])
        for ndx in range(self.conf.batch_size):
            for i_ndx in range(self.info[ndx][2][0]):
                cur_l = PoseTools.create_label_images(
                    self.locs[ndx:ndx+1,i_ndx,...], self.conf.imsz, 1, self.conf.label_blur_rad)
                label_ims[ndx,...] = np.maximum(label_ims[ndx,...], cur_l)

        self.fd[self.ph['y']] = label_ims

    def create_cursors(self, sess):
        PoseCommon.PoseCommonMulti.create_cursors(self,sess)


class PoseUNetTime(PoseUNet):
    def __init__(self,conf,name='pose_unet_time'):
        PoseUNet.__init__(self,conf.name)
        self.net_name = 'pose_unet_time'

    def create_ph(self):
        PoseCommon.PoseCommon.create_ph(self)
        imsz = self.conf.imsz
        rescale = self.conf.unet_rescale
        b_sz = self.conf.batch_size
        t_sz = self.conf.time_window_size*2 +1
        self.ph['x'] = tf.placeholder(tf.float32,
                           [b_sz*t_sz,imsz[0]/rescale,imsz[1]/rescale, self.conf.imgDim],
                           name='x')
        self.ph['y'] = tf.placeholder(tf.float32,
                           [b_sz,imsz[0]/rescale,imsz[1]/rescale, self.conf.n_classes],
                           name='y')
        self.ph['keep_prob'] = tf.placeholder(tf.float32)

    def create_fd(self):
        b_sz = self.conf.batch_size
        t_sz = self.conf.time_window_size*2 +1
        x_shape = [b_sz*t_sz,] + self.ph['x'].get_shape().as_list()[1:]
        y_shape = [b_sz,] + self.ph['y'].get_shape().as_list()[1:]
        self.fd = {self.ph['x']:np.zeros(x_shape),
                   self.ph['y']:np.zeros(y_shape),
                   self.ph['phase_train']:False,
                   self.ph['learning_rate']:0.
                   }

    def create_network1(self):
        m_sz = min(self.conf.imsz)/self.conf.unet_rescale
        max_layers = int(math.ceil(math.log(m_sz,2)))-1
        sel_sz = self.conf.sel_sz
        n_layers = int(math.ceil(math.log(sel_sz,2)))+2
        # max_layers = int(math.floor(math.log(m_sz)))
        # sel_sz = self.conf.sel_sz
        # n_layers = int(math.ceil(math.log(sel_sz)))+2
        n_layers = min(max_layers,n_layers) - 2

        n_conv = 2
        conv = PoseCommon.conv_relu3
        layers = []
        up_layers = []
        layers_sz = []
        X = self.ph['x']
        n_out = self.conf.n_classes
        debug_layers = []

        # downsample
        for ndx in range(n_layers):
            if ndx is 0:
                n_filt = 64
            elif ndx is 1:
                n_filt = 128
            elif ndx is 2:
                n_filt = 256
            else:
                n_filt = 512

            for cndx in range(n_conv):
                sc_name = 'layerdown_{}_{}'.format(ndx,cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
                debug_layers.append(X)
            layers.append(X)
            layers_sz.append(X.get_shape().as_list()[1:3])
            X = tf.nn.avg_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME')

        self.down_layers = layers
        self.debug_layers = debug_layers
        for cndx in range(n_conv):
            sc_name = 'layer_{}_{}'.format(n_layers,cndx)
            with tf.variable_scope(sc_name):
                X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])

        # upsample
        for ndx in reversed(range(n_layers)):
            if ndx is 0:
                n_filt = 64
            elif ndx is 1:
                n_filt = 128
            elif ndx is 2:
                n_filt = 256
            else:
                n_filt = 512
            X = CNB.upscale('u_'.format(ndx), X, layers_sz[ndx])

            if ndx>1:
            # rotate X along axis-0 and concat to provide context context along previous time steps.
                X_prev = []
                X_next = []
                for t in range(self.conf.time_window_size):
                    if not X_prev:
                        X_prev_cur = tf.concat([X[1:,...],X[0,...]],axis=0)
                        X_prev.append(X_prev_cur)
                        X_next_cur = tf.concat([X[-1,...],X[:-1,...]],axis=0)
                        X_next.append(X_next_cur)
                    else:
                        Z = X_prev[-1]
                        X_prev_cur = tf.concat([Z[1:,...],Z[0,...]],axis=0)
                        X_prev.append(X_prev_cur)
                        Z = X_next[-1]
                        X_next_cur = tf.concat([Z[-1,...],Z[:-1,...]],axis=0)
                        X_next.append(X_next_cur)

                X = tf.concat( X_next + [X, ]+ X_prev, axis = 3)

            X = tf.concat([X,layers[ndx]], axis=3)
            for cndx in range(n_conv):
                sc_name = 'layerup_{}_{}'.format(ndx, cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'], self.ph['keep_prob'])
            up_layers.append(X)
        self.up_layers = up_layers

        # final conv
        weights = tf.get_variable("out_weights", [3,3,n_filt,n_out],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("out_biases", n_out,
                                 initializer=tf.constant_initializer(0.))
        conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
        X = conv + biases

        t_sz = self.conf.time_window_size
        s_sz = 2*t_sz+1
        X = X[t_sz::s_sz,...]
        return X
