# ##
#
# import tensorflow as tf
# import PoseMDN
# from stephenHeadConfig import conf as conf
# import PoseTools
# import edward as ed
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import mymap
# import PoseTrain
# import os
# import pickle
# from tensorflow.contrib import slim
# from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
#
#
# tf.reset_default_graph()
#
# restore = True
# trainType = 0
# conf.batch_size = 16
# conf.nfcfilt = 256
# self = PoseTrain.PoseTrain(conf)
# self.createPH()
# self.createFeedDict()
# self.feed_dict[self.ph['keep_prob']] = 1.
# self.feed_dict[self.ph['phase_train_base']] = False
# self.feed_dict[self.ph['learning_rate']] = 0.
# self.trainType = 0
#
# with tf.variable_scope('base'):
#     self.createBaseNetwork(doBatch=True)
#
# self.conf.trange = self.conf.imsz[0] // 25
#
# self.openDBs()
# self.createBaseSaver()
#
# sess = tf.InteractiveSession()
# self.createCursors(sess)
# self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
# sess.run(tf.global_variables_initializer())
# self.restoreBase(sess, restore=True)
#
# #
#
# train_file = os.path.join(conf.cachedir,
#                          conf.trainfilename) + '.tfrecords'
# num_train = 0
# for t in tf.python_io.tf_record_iterator(train_file):
#     num_train += 1
#
# test_file = os.path.join(conf.cachedir,
#                          conf.valfilename) + '.tfrecords'
# num_test = 0
# for t in tf.python_io.tf_record_iterator(test_file):
#     num_test += 1
#
# train_data = []
# for ndx in range(num_train//conf.batch_size):
#     self.updateFeedDict(self.DBType.Train,distort=True,
#                         sess=sess)
#     cur_pred = sess.run(self.basePred, self.feed_dict)
#     train_data.append([cur_pred, self.locs, self.info, self.xs])
#
# test_data = []
# for ndx in range(num_test//conf.batch_size):
#     self.updateFeedDict(self.DBType.Val,distort=False,
#                         sess=sess)
#     cur_pred = sess.run(self.basePred, self.feed_dict)
#     test_data.append([cur_pred, self.locs, self.info, self.xs])
#
# with open(os.path.join(self.conf.cachedir, 'base_predictions'),
#           'wb') as f:
#     pickle.dump([train_data,test_data], f, protocol=2)


##

import tensorflow as tf
import PoseMDN
from stephenHeadConfig import conf as conf
import PoseTools
import edward as ed
import math
import numpy as np
import matplotlib.pyplot as plt
import mymap
import PoseTrain
import os
import pickle
from tensorflow.contrib import slim
from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
import copy

conf.mdn_min_sigma = 3.
conf.mdn_max_sigma = 4.
conf.expname = 'head_train_jitter_sigma3to4_my_bnorm_blocs'

# def create_local_network(self):
#
#     n_nets = 8
#     k = 4
#     conf = self.conf
#
#
#     l7_layer = self.ph['base_pred']
#     l7_shape = l7_layer.get_shape().as_list()
#     l7_layer_x_sz = l7_shape[2]
#     l7_layer_y_sz = l7_shape[1]
#
#     # self.ph['layer7'] = tf.placeholder(tf.float32, l7_shape)
#     # l7_layer = self.ph['layer7']
#
#     x_pad = (-l7_layer_x_sz) % n_nets
#     y_pad = (-l7_layer_y_sz) % n_nets
#     pad = [[0, 0], [0, y_pad], [0, x_pad], [0, 0]]
#     l7_layer = tf.pad(l7_layer,pad)
#     l7_shape = l7_layer.get_shape().as_list()
#     l7_layer_x_sz = l7_shape[2]
#     l7_layer_y_sz = l7_shape[1]
#
#     all_locs = []
#     all_scales = []
#     all_logits = []
#
#     x_psz = l7_layer_x_sz / n_nets
#     y_psz = l7_layer_y_sz / n_nets
#     assert x_psz == y_psz
#     with tf.variable_scope('mdn'):
#         for xx in range(n_nets):
#             for yy in range(n_nets):
#                 x_start = int(xx * x_psz)
#                 x_stop = int((xx + 1) * x_psz)
#                 y_start = int(yy * y_psz)
#                 y_stop = int((yy + 1) * y_psz)
#                 cur_patch = l7_layer[:, y_start:y_stop, x_start:x_stop, :]
#                 with tf.variable_scope('nn') as scope:
#                     if xx or yy:
#                         scope.reuse_variables()
#                         locs, scales, logits, hidden1 = self.mdn_network(
#                             cur_patch, k, conf.n_classes,
#                             np.array([x_psz]).astype('float32'))
#                     else:
#                         locs, scales, logits, hidden1 = self.mdn_network(
#                             cur_patch, k, conf.n_classes,
#                             np.array([x_psz]).astype('float32'))
#
#                 locs += np.array([x_start, y_start])
#                 all_locs.append(locs)
#                 all_scales.append(scales)
#                 all_logits.append(logits)
#         all_locs = tf.concat(all_locs, 1)
#         all_scales = tf.concat(all_scales, 1)
#         all_logits = tf.concat(all_logits, 1)
#     self.mdn_locs = all_locs
#     self.mdn_scales = all_scales
#     self.mdn_logits = all_logits
#
#     y_cls = []
#     for cls in range(self.conf.n_classes):
#         cat = Categorical(logits=all_logits[:, :, cls])
#         components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
#                       in zip(tf.unstack(tf.transpose(all_locs[:, :, cls, :], [1, 0, 2])),
#                              tf.unstack(tf.transpose(all_scales[:, :, cls, :], [1, 0, 2])))]
#         temp_tensor = tf.zeros([self.conf.batch_size, 2]) # not sure what this stupidity is
#         y_cls.append(Mixture(cat=cat, components=components,
#                              value=tf.zeros_like(temp_tensor)))
#
#     return y_cls

tf.reset_default_graph()
restore = True
trainType = 0
conf.batch_size = 16
conf.nfcfilt = 256
self = PoseMDN.PoseMDN(conf)

mdn_dropout = 0.95
self.create_ph()
self.createFeedDict()
self.feed_dict[self.ph['keep_prob']] = mdn_dropout
self.feed_dict[self.ph['phase_train_base']] = False
self.feed_dict[self.ph['phase_train_mdn']] = True
self.feed_dict[self.ph['learning_rate']] = 0.
self.trainType = trainType

self.ph['base_pred'] = tf.placeholder(
    tf.float32, [None, 128, 128, self.conf.n_classes])

y = self.create_network(self.ph['base_pred'])
self.mdn_out = y
self.openDBs()
self.create_saver()

y_label = self.ph['base_locs'] / self.conf.rescale / self.conf.pool_scale
self.mdn_label = y_label
data_dict = {}
for ndx in range(self.conf.n_classes):
    data_dict[y[ndx]] = y_label[:, ndx, :]
inference = mymap.MAP(data=data_dict)
inference.initialize(var_list=PoseTools.getvars('mdn'))
self.loss = inference.loss


# # old settings
starter_learning_rate = 0.0001
decay_steps = 5000 * 8 / self.conf.batch_size
learning_rate = tf.train.exponential_decay(
    starter_learning_rate, self.ph['step'], decay_steps, 0.9,
    staircase=True)

# # new settings
# starter_learning_rate = 0.0001
# decay_steps = 12000 * 8 / self.conf.batch_size
# learning_rate = tf.train.exponential_decay(
#     starter_learning_rate, self.ph['step'], decay_steps, 0.9)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    self.opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(self.loss)

# adam_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
# self.opt =slim.learning.create_train_op(
#     self.loss, adam_opt, global_step=step)
sess = tf.Session()

self.createCursors(sess)
self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                                 self.conf.n_classes, 2])
sess.run(tf.global_variables_initializer())
self.restore(sess, restore)
# self.initializeRemainingVars(sess)

with open(os.path.join(conf.cachedir, 'base_predictions'),
          'rb') as f:
    train_data, test_data = pickle.load(f)

m_train_data = train_data
o_test_data = copy.deepcopy(test_data)
for ndx in range(len(test_data)):
    bpred_in = test_data[ndx][0]
    tr_ndx = np.random.choice(len(train_data))
    bpred_tr = train_data[tr_ndx][0]
    bpred_out = -1 * np.ones(bpred_in.shape)
    tr_locs = PoseTools.getBasePredLocs(bpred_tr, self.conf)/4
    in_locs = PoseTools.getBasePredLocs(bpred_in, self.conf)/4

    # training blobs test locations
    # for ex in range(bpred_in.shape[0]):
    #     for cls in range(bpred_in.shape[-1]):
    #         cur_sc = bpred_in[ex,:,:,cls]
    #         inxx = int(in_locs[ex,cls,0])
    #         inyy = int(in_locs[ex,cls,1])
    #         trxx = int(tr_locs[ex,cls,0])
    #         tryy = int(tr_locs[ex,cls,1])
    #         tr_sc = bpred_tr[ex,(tryy-10):(tryy+10),
    #                 (trxx - 10):(trxx + 10),cls]
    #         bpred_out[ex,(inyy-10):(inyy+10),
    #             (inxx - 10):(inxx + 10),cls ] = tr_sc

    # test blobs training locs
    for ex in range(bpred_in.shape[0]):
        for cls in range(bpred_in.shape[-1]):
            cur_sc = bpred_in[ex,:,:,cls]
            inxx = int(in_locs[ex,cls,0])
            inyy = int(in_locs[ex,cls,1])
            trxx = int(tr_locs[ex,cls,0])
            tryy = int(tr_locs[ex,cls,1])
            test_sc = bpred_in[ex,(inyy-10):(inyy+10),
                    (inxx - 10):(inxx + 10),cls]
            bpred_out[ex,(tryy-10):(tryy+10),
                (trxx - 10):(trxx + 10),cls ] = test_sc
    test_data[ndx][0] = bpred_out
# m_test_data = [i for sl in zip(test_data,o_test_data) for i in sl]
m_test_data = o_test_data

##

self.updateFeedDict(self.DBType.Train, sess=sess,
                    distort=True)

mdn_steps = 50000 * 8 / self.conf.batch_size
test_step = 0
self.feed_dict[self.ph['phase_train_mdn']] = True
for cur_step in range(self.mdn_start_at, mdn_steps):
    self.feed_dict[self.ph['step']] = cur_step
    data_ndx = cur_step%len(m_train_data)
    cur_bpred = m_train_data[data_ndx][0]
    pd = 15
    cur_bpred = np.pad(cur_bpred,[[0,0],[pd,pd],[pd,pd],[0,0]],
                       'constant',constant_values=-1)
    dxx = np.random.randint(pd*2)
    dyy = np.random.randint(pd*2)
    cur_bpred = cur_bpred[:,dyy:(128+dyy),dxx:(128+dxx),:]
    # self.feed_dict[self.ph['step']] = cur_step
    self.feed_dict[self.ph['base_locs']] = \
        PoseTools.getBasePredLocs(cur_bpred, self.conf)
    self.feed_dict[self.ph['base_pred']] = cur_bpred
    sess.run(self.opt, self.feed_dict)

    if cur_step % self.conf.display_step == 0:
        data_ndx = (cur_step+1) % len(m_train_data)
        cur_bpred = m_train_data[data_ndx][0]
        self.feed_dict[self.ph['base_locs']] = \
            PoseTools.getBasePredLocs(cur_bpred, self.conf)
        self.feed_dict[self.ph['base_pred']] = cur_bpred
        tr_loss = sess.run(self.loss, feed_dict=self.feed_dict)
        self.mdn_train_data['train_err'].append(tr_loss)
        self.mdn_train_data['step_no'].append(cur_step)

        mdn_pred = self.mdn_pred(sess)
        bee = PoseTools.getBaseError(
            self.feed_dict[self.ph['base_locs']], mdn_pred, conf)
        tt1 = np.sqrt(np.sum(np.square(bee), 2))
        nantt1 = np.invert(np.isnan(tt1.flatten()))
        train_dist = tt1.flatten()[nantt1].mean()
        self.mdn_train_data['train_dist'].append(train_dist)

        data_ndx = (test_step + 1) % len(m_test_data)
        cur_bpred = m_test_data[data_ndx][0]
        self.feed_dict[self.ph['base_locs']] = \
            PoseTools.getBasePredLocs(cur_bpred, self.conf)
        self.feed_dict[self.ph['base_pred']] = cur_bpred
        cur_te_loss = sess.run(self.loss, feed_dict=self.feed_dict)
        val_loss = cur_te_loss
        mdn_pred = self.mdn_pred(sess)
        bee = PoseTools.getBaseError(
            self.feed_dict[self.ph['base_locs']],mdn_pred,conf)
        tt1 = np.sqrt(np.sum(np.square(bee), 2))
        nantt1 = np.invert(np.isnan(tt1.flatten()))
        val_dist = tt1.flatten()[nantt1].mean()
        test_step += 1

        self.mdn_train_data['val_err'].append(val_loss)
        self.mdn_train_data['val_dist'].append(val_dist)

        print('{}:Train Loss:{:.4f},{:.2f}, Test Loss:{:.4f},{:.2f}'.format(
            cur_step, tr_loss, train_dist, val_loss, val_dist))

    if cur_step % self.conf.save_step == 0:
        self.save(sess, cur_step)

print("Optimization finished!")
self.save(sess, mdn_steps)
self.closeCursors()


##
