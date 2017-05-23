from PoseTrain import PoseTrain
import os
import sys
import tensorflow as tf
import PoseTools
import re
import pickle
from tensorflow.contrib import slim
from edward.models import Categorical, Mixture, Normal

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

        pred,layers=super(self.__class__, self).createBaseNetwork(doBatch=True)
        self.basePred = pred
        self.baseLayers = layers

        l7layer = layers['conv7']
        with tf.variable_scope('mdn'):
        # l7_sh = tf.shape(l7layer)
            l7layer_re = tf.stop_gradient(tf.reshape(l7layer, [self.conf.batch_size,-1]))
            mdn_hidden1 = slim.fully_connected(l7layer_re, 256)
            mdn_hidden2 = slim.fully_connected(mdn_hidden1, 256)
            locs = slim.fully_connected(mdn_hidden2, K, activation_fn=None)
            scales = slim.fully_connected(mdn_hidden2, K, activation_fn=tf.exp)
            logits = slim.fully_connected(mdn_hidden2, K, activation_fn=None)

        cat = Categorical(logits=logits)
        components = [Normal(locs=loc,scales=scale) for loc, scale
                      in zip(tf.unstack(tf.transpose(locs)),
                             tf.unstack(tf.transpose(scales)))]
        y = Mixture(cat=cat, components=components, value=self.ph['locs'])

        self.mdn_locs = locs
        self.mdn_scales = scales
        self.mdn_logits = logits
        self.mdn_y = y


    def create_ph(self):
        super(self.__class__, self).createPH()


    def train(self, restore, trainType):
        PoseTrain = super(self.__class__,self)
        self.create_ph()
        self.createFeedDict()
        self.feed_dict[self.ph['keep_prob']] = 0.5
        self.feed_dict[self.ph['phase_train_base']] = False
        self.trainType = trainType


        with tf.session() as sess:
            PoseTrain.restoreBase(sess, restore=True)