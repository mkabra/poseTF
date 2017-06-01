from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object

from PoseTrain import PoseTrain
import os
import sys
import tensorflow as tf
import PoseTools
import re
import pickle
from tensorflow.contrib import slim
from edward.models import Categorical, Mixture, Normal
import edward as ed
import math

class PoseMDN(PoseTrain):

    def __init__(self, conf):
        super(self.__class__, self).__init__(conf)
        self.mdn_saver = None
        self.mdn_start_at = 0
        self.mdn_train_data = {}
        self.baseLayers = {}
        self.basePred = None
        self.mdn_locs = None
        self.mdn_scales = None
        self.mdn_logits = None

    def create_saver(self):
        self.mdn_saver = tf.train.Saver(var_list = PoseTools.getvars('mdn'),
                                        max_to_keep=self.conf.maxckpt)

    def restore(self, sess, restore):
        out_filename = os.path.join(self.conf.cachedir,self.conf.expname+ '_MDN')
        out_filename = out_filename.replace('\\', '/')
        train_data_filename = os.path.join(self.conf.cachedir,self.conf.expname + '_MDN_traindata')
        ckpt_filename = self.conf.expname + '_MDN_ckpt'
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                            latest_filename = ckpt_filename)
        if not latest_ckpt or not restore:
            self.mdn_start_at = 0
            self.mdn_train_data = {'train_err':[], 'val_err':[], 'step_no':[],
                                  'train_dist':[], 'val_dist':[] }
            sess.run(tf.variables_initializer(PoseTools.getvars('mdn')),feed_dict=self.feed_dict)
            print("Not loading MDN variables. Initializing them")
            return False
        else:
            self.mdn_saver.restore(sess,latest_ckpt.model_checkpoint_path)
            match_obj = re.match(out_filename + '-(\d*)',latest_ckpt.model_checkpoint_path)
            self.mdn_start_at = int(match_obj.group(1))+1
            with open(train_data_filename,'rb') as td_file:
                if sys.version_info.major==3:
                    in_data = pickle.load(td_file,encoding='latin1')
                else:
                    in_data = pickle.load(td_file)

                if not isinstance(in_data,dict):
                    self.mdn_train_data, load_conf = in_data
                    print('Parameters that dont match for base:')
                    PoseTools.compareConf(self.conf, load_conf)
                else:
                    print("No config was stored for base. Not comparing conf")
                    self.mdn_train_data = in_data
            print("Loading MDN variables from %s"%latest_ckpt.model_checkpoint_path)
            return True

    def save(self, sess, step):
        out_filename = os.path.join(self.conf.cachedir,self.conf.expname+ '_MDN')
        out_filename = out_filename.replace('\\', '/')
        train_data_filename = os.path.join(self.conf.cachedir,self.conf.expname + '_MDN_traindata')
        ckpt_filename = self.conf.expname + '_MDN_ckpt'
        self.mdn_saver.save(sess,out_filename,global_step=step,
                   latest_filename=ckpt_filename)
        print('Saved state to %s-%d' %(out_filename,step))
        with open(train_data_filename,'wb') as td_file:
            pickle.dump([self.basetrainData,self.conf],td_file,protocol=2)

    def create_network(self):
        K = 100

        with tf.variable_scope('base'):
            super(self.__class__, self).createBaseNetwork(doBatch=True)

        l7layer = self.baseLayers['conv7']
        n_out = K*self.conf.n_classes*2
        with tf.variable_scope('mdn'):
            # l7_sh = l7layer.get_shape().as_list()
            # l7_sz = l7_sh[1]*l7_sh[2]*l7_sh[3]
            # l7layer_re = tf.reshape(l7layer, [self.conf.batch_size, l7_sz])
            l7layer_re = tf.stop_gradient(slim.flatten(l7layer))
            mdn_hidden1 = slim.fully_connected(l7layer_re, 256)
            mdn_hidden2 = slim.fully_connected(mdn_hidden1, 256)
            locs = slim.fully_connected(mdn_hidden2, n_out, activation_fn=None)
            scales = slim.fully_connected(mdn_hidden2, n_out, activation_fn=tf.exp)
            logits = slim.fully_connected(mdn_hidden2, n_out, activation_fn=None)

        locs = tf.reshape(locs, [-1,self.conf.n_classes,2,K])
        scales = tf.reshape(scales, [-1,self.conf.n_classes,2,K])
        logits = tf.reshape(logits, [-1,self.conf.n_classes,2,K])
        y_stack = []
        loss = None
        for dim in range(2):
            for cndx in range(self.conf.n_classes):
                cat = Categorical(logits=logits[:,cndx,dim,:])
                components = [Normal(loc=loc,scale=scale) for loc, scale
                              in zip(tf.unstack(tf.transpose(locs[:,cndx,dim,:])),
                                     tf.unstack(tf.transpose(scales[:,cndx,dim,:])))]
                cur_y = Mixture(cat=cat, components=components, value=tf.zeros_like(self.ph['locs'][:,0,0]))
                curinference = ed.MAP(data={cur_y:self.ph['locs'][:,cndx,dim]})
                curinference.initialize(var_list=PoseTools.getvars('mdn'))
                if loss is None:
                    loss = curinference.loss
                else:
                    loss += curinference.loss
        self.mdn_locs = locs
        self.mdn_scales = scales
        self.mdn_logits = logits
        self.loss = loss


    def create_ph(self):
        super(self.__class__, self).createPH()


    def train(self, restore, trainType):
        self.conf.psz = 4

        mdn_dropout = 0.5
        self.create_ph()
        self.createFeedDict()
        self.feed_dict[self.ph['keep_prob']] = mdn_dropout
        self.feed_dict[self.ph['phase_train_base']] = False
        self.trainType = trainType

        self.create_network()
        self.openDBs()
        self.createBaseSaver()

        self.opt = tf.train.AdamOptimizer(learning_rate= self.ph['learning_rate']).minimize(self.loss)

        with tf.Session() as sess:
            self.createCursors(sess)
            self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
            self.restoreBase(sess, restore=True)
            for dim in range(2):
                for cls in range(self.conf.n_classes):
                    inference[dim,cls].initialize(var_list=PoseTools.getvars('mdn'))

            self.initializeRemainingVars(sess)
            for step in range(1000):
                self.updateFeedDict(self.DBType.Train, sess=sess,
                                    distort=True)
                self.feed_dict[self.ph['keep_prob']] = mdn_dropout
                ex_count = step * self.conf.batch_size
                cur_lr = self.conf.base_learning_rate * \
                         self.conf.gamma ** math.floor(ex_count/ self.conf.step_size)
                self.feed_dict[self.ph['learning_rate']] = cur_lr
                sess.run(self.opt,self.feed_dict)

                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train, sess=sess,
                                        distort=True)
                    self.feed_dict[self.ph['keep_prob']] = 1.
                    tr_loss = sess.run(loss,feed_dict=self.feed_dict)

                    numrep = int(self.conf.numTest/self.conf.batch_size) + 1
                    te_loss = 0.
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val, sess=sess,
                                            distort=False)
                        self.feed_dict[self.ph['keep_prob']] = 1.
                        cur_te_loss = sess.run(loss, feed_dict=self.feed_dict)
                        te_loss += cur_te_loss
                    te_loss /= numrep

                    print('Train Loss:{:.4f}, Test Loss:{:.4f}'.format(tr_loss,te_loss))

                if step % conf.save_step == 0:
                    self.save(sess,step)

            print("Optimization finished!")
            self.save(sess,step)
        self.closeCursors()