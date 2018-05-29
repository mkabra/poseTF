# ## testing batch norm
# import tensorflow as tf
# from tensorflow.contrib.layers import batch_norm
# import numpy as np
# #from batch_norm import batch_norm_new as batch_norm
#
# tf.reset_default_graph()
#
# # Two layer network.
# phase_train_mdn = tf.placeholder(tf.bool)
#
# x_flat = tf.placeholder(tf.float32,[3,5])
# weights1 = tf.get_variable("weights1", [5, 3],
#                            initializer=tf.random_normal_initializer())
# biases1 = tf.get_variable("biases1", 1,
#                           initializer=tf.constant_initializer(0))
# hidden1_b = tf.matmul(x_flat, weights1)
# hidden1_a = batch_norm(hidden1_b, decay=0.8,
#                      is_training=phase_train_mdn,updates_collections=None)
# hidden1 = hidden1_a + biases1
# # without +1 all hidden1 end up being zeros.
#
# weights2 = tf.get_variable("weights2", [3, 1],
#                            initializer=tf.contrib.layers.xavier_initializer())
# biases2 = tf.get_variable("biases2", 1,
#                           initializer=tf.constant_initializer(0))
# hidden2 = tf.matmul(hidden1, weights2) + biases2
#
# y = tf.placeholder(tf.float32,[3,])
# step = tf.placeholder(tf.int32)
#
# loss = tf.nn.l2_loss(hidden2-y)
# ##
#
# learning_rate = tf.train.exponential_decay(
#     0.1, step, 100, 0.9)
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
# sess = tf.InteractiveSession()
# x_in = np.random.rand(3, 5)
# y_in = np.random.rand(3)
# sess.run(tf.variables_initializer(tf.global_variables()),
#          feed_dict={x_flat:x_in,y:y_in,step:0})
# for ndx in range(100):
#     x_in = np.random.rand(3,5)
#     y_in = np.random.rand(3)
#     sess.run(opt,feed_dict={x_flat:x_in,y:y_in,
#                             step:0,phase_train_mdn:True})
#
# x_in1 = np.random.rand(3,5)
#
# ##
# y_1,yy_1 = sess.run([hidden1,hidden1_b],
#                     feed_dict={x_flat:x_in1,y:y_in,
#                                step:0,phase_train_mdn:False})
# y_2,yy_2 = sess.run([hidden1,hidden1_b],
#                     feed_dict={x_flat:x_in1,y:y_in,
#                                step:0,phase_train_mdn:False})
# y_3,yy_3 = sess.run([hidden1,hidden1_b],
#                     feed_dict={x_flat:x_in1,y:y_in,
#                                step:0,phase_train_mdn:True})
# y_4,yy_4 = sess.run([hidden1,hidden1_b],
#                     feed_dict={x_flat:x_in1,y:y_in,
#                                step:0,phase_train_mdn:True})
# y_5,yy_5 = sess.run([hidden1,hidden1_b],
#                     feed_dict={x_flat:x_in1,y:y_in,
#                                step:0,phase_train_mdn:False})
# y_6,yy_6 = sess.run([hidden1,hidden1_b],
#                     feed_dict={x_flat:x_in1,y:y_in,
#                                step:0,phase_train_mdn:True})
# print y_1
# print y_2
# print y_3
# print y_4
# print y_5
# print y_6
#

#
## cv for stephen.

import copy
import tensorflow as tf
import PoseUNet
import os
import numpy as np
import PoseTools
tf.reset_default_graph()

split_type = 'easy'
# extra_str = '_normal_bnorm'
extra_str = '_my_bnorm'
imdim = 1

def get_cache_dir(conf, split, split_type):
    outdir = os.path.join(conf.cachedir, 'cv_split_{}'.format(split))
    if split_type is 'easy':
        outdir += '_easy'
    outdir += extra_str
    return outdir

view = 1
split_num = 1
if view == 0:
    from stephenHeadConfig import sideconf as conf
    curconf = copy.deepcopy(conf)
    curconf.imsz = (350, 230)
else:
    from stephenHeadConfig import conf as conf
    curconf = copy.deepcopy(conf)
    curconf.imsz = (350, 350)
curconf.batch_size = 4
curconf.unet_rescale = 1
curconf.cachedir = get_cache_dir(curconf, split_num, split_type)
curconf.imgDim = 1

tf.reset_default_graph()
self = PoseUNet.PoseUNet(curconf)
# dd,ims,preds,pred_locs,locs,info = self.classify_val(0)
# print(np.percentile(dd,[90,95,98],axis=0))

self.init_train(0)
self.pred = self.create_network()
self.cost = tf.nn.l2_loss(self.pred - self.ph['y'])
self.create_saver()

sess = tf.Session()

td_fields = ('loss','dist')
self.init_and_restore(sess,True,td_fields)
vv = tf.global_variables()

td = 0
nn = 377
info = []
dist = []
for idx in range(nn):
    self.setup_val(sess)
    self.fd[self.ph['keep_prob']] = 1.
    self.fd[self.ph['phase_train']] = False
    cur_pred = sess.run(self.pred, self.fd)
    cur_locs = PoseTools.get_pred_locs(cur_pred)
    tt = cur_locs - self.locs
    tt1 = np.sqrt(np.sum(tt**2,2))
    dist.extend(tt1)
    info.extend(self.info)
    train_dist = self.compute_dist(cur_pred, self.locs)
    td += train_dist
    if (idx+1)%50==0:
        print('.')
print(td/nn)
dist = np.array(dist)
info = np.array(info)
print(np.percentile(dist,[90,95,98],axis=0))
print(info[:10,:])

info = []
dist = []
for idx in range(nn):
    self.setup_train(sess)
    self.fd[self.ph['keep_prob']] = 1.
    self.fd[self.ph['phase_train']] = True
    cur_pred = sess.run(self.pred, self.fd)
    cur_locs = PoseTools.get_pred_locs(cur_pred)
    tt = cur_locs - self.locs
    tt1 = np.sqrt(np.sum(tt**2,2))
    dist.extend(tt1)
    info.extend(self.info)
    train_dist = self.compute_dist(cur_pred, self.locs)
    td += train_dist
    if (idx+1)%50==0:
        print('.')
print(td/nn)
dist = np.array(dist)
info = np.array(info)
print(np.percentile(dist,[90,95,98],axis=0))
print(info[:10,:])


# import PoseUNet
# from poseConfig import aliceConfig as conf
# import APT_interface
# import PoseTools
# import tensorflow as tf
# import os
# import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# tf.reset_default_graph()
# dirs = ['/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00006_CsChr_RigC_20151014T093157',
# '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00217_CsChr_RigB_20150929T095409',
# '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00277_CsChr_RigD_20150819T094557',
# '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_JRC_SS03500_CsChr_RigB_20150811T114536',
# '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00243_CsChr_RigD_20150812T155340']
#
# self = PoseUNet.PoseUNet(conf, name='pose_unet_full_20180521')
# sess = self.init_net_meta(1, True)
#
# def pred_fn(all_f):
#     bsize = conf.batch_size
#     xs, _ = PoseTools.preprocess_ims(
#         all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]), conf=self.conf,
#         distort=False, scale=self.conf.unet_rescale)
#
#     self.fd[self.ph['x']] = xs
#     self.fd[self.ph['phase_train']] = False
#     self.fd[self.ph['keep_prob']] = 1.
#     pred = sess.run(self.pred, self.fd)
#     base_locs = PoseTools.get_pred_locs(pred)
#     base_locs = base_locs * conf.unet_rescale
#     return base_locs
#
# for bdir in dirs:
#     mov = os.path.join(bdir,'movie.ufmf')
#     trx = os.path.join(bdir,'registered_trx.mat')
#     out_file = os.path.join(bdir,'movie_unet_20180512.trk')
#     APT_interface.classify_movie(conf,pred_fn, mov, out_file, trx)
#

# import PoseUNetAttention
# import tensorflow as tf
# from poseConfig import aliceConfig as conf
#
# self = PoseUNetAttention.PoseUNetAttention(conf,)