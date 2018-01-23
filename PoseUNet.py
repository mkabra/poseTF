import PoseCommon
import PoseTools
import tensorflow as tf
import os
import sys
import math
from tensorflow.contrib.layers import batch_norm
import convNetBase as CNB
import numpy as np
import cv2
from cvc import cvc
from PoseTools import scale_images

class PoseUNet(PoseCommon.PoseCommon):

    def __init__(self, conf, name='pose_unet'):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.down_layers = [] # layers created while down sampling
        self.up_layers = [] # layers created while up sampling
        self.edge_ignore = 10
        self.net_name = 'pose_unet'

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

        n_conv = 3
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
                    X = conv(X, n_filt, self.ph['phase_train'])
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
                X = conv(X, n_filt, self.ph['phase_train'])

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
                    X = conv(X, n_filt, self.ph['phase_train'])
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

    def fd_train(self):
        self.fd[self.ph['phase_train']] = True

    def fd_val(self):
        self.fd[self.ph['phase_train']] = False

    def update_fd(self, db_type, sess, distort):
        self.read_images(db_type, distort, sess, distort)
        rescale = self.conf.unet_rescale
        xs = scale_images(self.xs, rescale, self.conf)
        self.fd[self.ph['x']] = PoseTools.normalize_mean(xs, self.conf)
        imsz = [self.conf.imsz[0]/rescale, self.conf.imsz[1]/rescale,]
        label_ims = PoseTools.create_label_images(
            self.locs/rescale, imsz, 1, self.conf.label_blur_rad)
        self.fd[self.ph['y']] = label_ims


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

        return val_dist, val_ims, val_preds, val_predlocs, val_locs/self.conf.unet_rescale

    def classify_movie(self, movie_name, sess, max_frames=-1, start_at=0):
        # maxframes if specificied reads that many frames
        # start at specifies where to start reading.
        conf = self.conf

        cap = cv2.VideoCapture(movie_name)
        n_frames = int(cap.get(cvc.FRAME_COUNT))

        # figure out how many frames to read
        if max_frames > 0:
            if max_frames + start_at > n_frames:
                n_frames = n_frames - start_at
            else:
                n_frames = max_frames
        else:
            n_frames = n_frames - start_at

        # since out of order frame access in python sucks, do it one by one
        for _ in range(start_at):
            cap.read()

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
                success, frame_in = cap.read()
                assert success, "Could not read frame"
                frame_in = PoseTools.crop_images(frame_in, conf)
                all_f[ii, ...] = frame_in[..., 0:1]

            x0 = scale_images(all_f, conf.unet_rescale, conf)
            self.fd[self.ph['x']] = x0
            self.fd[self.ph['phase_train']] = False
            pred = sess.run(self.pred, self.fd)

            base_locs = PoseTools.get_base_pred_locs(pred, conf)
            pred_locs[ndx_start:ndx_end, :, :] = base_locs[:ppe, :, :]
            pred_max_scores[ndx_start:ndx_end, :] = pred[:ppe, :, :, :].max(axis=(1,2))
            pred_scores[ndx_start:ndx_end, :, :, :] = pred[:ppe, :, :, :]
            sys.stdout.write('.')
            if curl % 20 == 19:
                sys.stdout.write('\n')

        cap.release()
        return pred_locs, pred_scores, pred_max_scores

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
