import PoseCommon
import PoseTools
import tensorflow as tf
import os
import sys
import math
from tensorflow.contrib.layers import batch_norm
import convNetBase as CNB
import numpy as np

class PoseUNet(PoseCommon.PoseCommon):

    def __init__(self, conf, name='pose_unet'):
        PoseCommon.PoseCommon.__init__(self, conf, name)
        self.down_layers = [] # layers created while down sampling
        self.up_layers = [] # layers created while up sampling
        self.edge_ignore = 10

    def create_ph(self):
        PoseCommon.PoseCommon.create_ph(self)
        imsz = self.conf.imsz
        self.ph['x'] = tf.placeholder(tf.float32,
                           [None,imsz[0],imsz[1], self.conf.imgDim],
                           name='x')
        self.ph['y'] = tf.placeholder(tf.float32,
                           [None,imsz[0],imsz[1], self.conf.n_classes],
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
        with tf.variable_scope(self.name):
            return self.create_network1()

    def create_network1(self):
        m_sz = min(self.conf.imsz)
        max_layers = int(math.floor(math.log(m_sz)))
        sel_sz = self.conf.sel_sz
        n_layers = int(math.ceil(math.log(sel_sz)))+2
        n_layers = min(max_layers,n_layers)

        n_conv = 3
        conv = PoseCommon.conv_relu3
        layers = []
        up_layers = []
        layers_sz = []
        X = self.ph['x']
        n_out = self.conf.n_classes

        # downsample
        for ndx in range(n_layers):
            if ndx is 0:
                n_filt = 32
            elif ndx is 1:
                n_filt = 64
            # elif ndx is 2:
            #     n_filt = 256
            else:
                n_filt = 128

            for cndx in range(n_conv):
                sc_name = 'layerdown_{}_{}'.format(ndx,cndx)
                with tf.variable_scope(sc_name):
                    X = conv(X, n_filt, self.ph['phase_train'])

            layers.append(X)
            layers_sz.append(X.get_shape().as_list()[1:3])
            X = tf.nn.max_pool(X,ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME')

        self.down_layers = layers
        # few more convolution for the final layers
        for cndx in range(n_conv):
            sc_name = 'layer_{}_{}'.format(n_layers,cndx)
            with tf.variable_scope(sc_name):
                X = conv(X, n_filt, self.ph['phase_train'])

        # upsample
        for ndx in reversed(range(n_layers)):
            if ndx is 0:
                n_filt = 32
            elif ndx is 1:
                n_filt = 64
            # elif ndx is 2:
            #     n_filt = 256
            else:
                n_filt = 128
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
        self.read_images(db_type, distort, sess)
        self.fd[self.ph['x']] = self.xs
        label_ims = PoseTools.create_label_images(
            self.locs, self.conf.imsz, 1, self.conf.label_blur_rad)
        self.fd[self.ph['y']] = label_ims


    def train_unet(self, restore, train_type=0):

        def loss(pred_in, pred_out):
            return tf.nn.l2_loss(pred_in-pred_out)

        PoseCommon.PoseCommon.train(self,
            restore=restore,
            train_type=train_type,
            create_network=self.create_network,
            training_iters=60000/self.conf.batch_size*8,
            loss=loss,
            pred_in_key='y',
            learning_rate=0.0001,
            td_fields=('loss','dist'))

    def classify_val(self, train_type=0):
        PoseCommon.PoseCommon.classify_val(self,train_type)


class PoseUNetMulti(PoseUNet, PoseCommon.PoseCommonMulti):

    def __init__(self, conf, name='pose_unet_multi'):
        PoseUNet.__init__(self, conf, name)

    def update_fd(self, db_type, sess, distort):
        self.read_images(db_type, distort, sess)
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
