from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object
import tensorflow as tf
import os
import PoseTools
import multiResData
from enum import Enum
import numpy as np
import re
import pickle
import sys
import math
from past.utils import old_div
from tensorflow.contrib.layers import batch_norm
# from batch_norm import batch_norm_mine as batch_norm
#from matplotlib import pyplot as plt
import copy
import cv2
import gc
import resource
import json


def conv_relu(x_in, kernel_shape, train_phase):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, train_phase)
    return tf.nn.relu(conv + biases)


def conv_relu3(x_in, n_filt, train_phase, keep_prob=None):
    in_dim = x_in.get_shape().as_list()[3]
    kernel_shape = [3, 3, in_dim, n_filt]
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, decay=0.99, is_training=train_phase)
    if keep_prob is not None:
        conv = tf.nn.dropout(conv, keep_prob)
    return tf.nn.relu(conv + biases)

def conv_relu3_noscaling(x_in, n_filt, train_phase):
    in_dim = x_in.get_shape().as_list()[3]
    kernel_shape = [3, 3, in_dim, n_filt]
    weights = tf.get_variable(
        "weights", kernel_shape,
        initializer=
        tf.contrib.layers.variance_scaling_initializer())
    biases = tf.get_variable("biases", kernel_shape[-1],
                             initializer=tf.constant_initializer(0.))
    conv = tf.nn.conv2d(x_in, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = batch_norm(conv, decay=0.99, is_training=train_phase)
    return tf.nn.relu(conv + biases)

def print_train_data(cur_dict):
    p_str = ''
    for k in cur_dict.keys():
        p_str += '{:s}:{:.2f} '.format(k, cur_dict[k])
    print(p_str)


def initialize_remaining_vars(sess):
    var_list = set(sess.run(tf.report_uninitialized_variables()))
    vlist = [v for v in tf.global_variables() if \
             v.name.split(':')[0] in var_list]
    sess.run(tf.variables_initializer(vlist))
    # at some stage do faster initialization.
    # var_list = tf.global_variables()
    # for var in var_list:
    #     # if ignore_name is not None and var.name.startswith(ignore_name):
    #     #     continue
    #     try:
    #         sess.run(tf.assert_variables_initialized([var]))
    #     except tf.errors.FailedPreconditionError:
    #         sess.run(tf.variables_initializer([var]))
    #         print('Initializing variable:%s' % var.name)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def l2_dist(x,y):
    return np.sqrt(np.sum((x-y)**2,axis=-1))

class PoseCommon(object):

    class DBType(Enum):
        Train = 1
        Val = 2

    def __init__(self, conf, name):
        self.coord = None
        self.threads = None
        self.conf = conf
        self.name = name
        self.net_name = 'pose_common'
        self.train_type = None
        self.train_data = None
        self.val_data = None
        self.train_queue = None
        self.val_queue = None
        self.xs = None
        self.locs = None
        self.info = None
        self.ph = {}
        self.fd = {}
        self.train_info = {}
        self.cost = None
        self.pred = None
        self.opt = None
        self.saver = None
        self.dep_nets = None
        self.joint = False
        self.edge_ignore = 0 # amount of edge to ignore while computing prediction locations
        self.train_loss = np.zeros(500) # keep track of past loss for importance sampling
        self.step = 0 # might be required for mdn.
        self.extra_data = None
        self.db_name = ''
        self.read_and_decode = multiResData.read_and_decode

    def open_dbs(self):
        assert self.train_type is not None, 'traintype has not been set'
        if self.train_type == 0:
            train_filename = os.path.join(self.conf.cachedir, self.conf.trainfilename) + self.db_name + '.tfrecords'
            val_filename = os.path.join(self.conf.cachedir, self.conf.valfilename) + self.db_name + '.tfrecords'
            self.train_queue = tf.train.string_input_producer([train_filename])
            self.val_queue = tf.train.string_input_producer([val_filename])
        else:
            train_filename = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename) + self.db_name + '.tfrecords'
            val_filename = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename) + self.db_name + '.tfrecords'
            self.train_queue = tf.train.string_input_producer([train_filename])
            self.val_queue = tf.train.string_input_producer([val_filename])

    def create_cursors(self, sess):
        tdata = self.read_and_decode(self.train_queue, self.conf)
        vdata = self.read_and_decode(self.val_queue, self.conf)
        self.train_data = tdata
        self.val_data = vdata
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def close_cursors(self):
        self.coord.request_stop()
        self.coord.join(self.threads)

    def read_images(self, db_type, distort, sess, shuffle=None, scale = 1):
        conf = self.conf
        cur_data = self.val_data if (db_type == self.DBType.Val)\
            else self.train_data
        xs = []
        locs = []
        info = []
        extra_data = []

        if shuffle is None:
            shuffle = distort

        # Tfrecords doesn't allow shuffling. Skipping a random
        # number of records
        # as a way to simulate shuffling. very hacky.

        for _ in range(conf.batch_size):
            if shuffle:
                for _ in range(np.random.randint(30)):
                    sess.run(cur_data)
            A = sess.run(cur_data)
            xs.append(A[0])
            locs.append(A[1])
            info.append(A[2])
            if len(A)>3:
                extra_data.append(A[3:])
        xs = np.array(xs)
        locs = np.array(locs)
        locs = multiResData.sanitize_locs(locs)

        xs, locs = PoseTools.preprocess_ims(xs, locs, conf, distort, scale)
        self.xs = xs
        self.locs = locs
        self.info = info
        self.extra_data = extra_data

    def create_saver(self):
        saver = {}
        name = self.name
        net_name = self.net_name
        saver['out_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name)
        saver['train_data_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name + '_traindata')
        saver['ckpt_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name + '_ckpt')
        saver['saver'] = (tf.train.Saver(var_list=PoseTools.get_vars(net_name+ '/'),
                                         max_to_keep=self.conf.maxckpt,
                                         save_relative_paths=True))
        self.saver = saver
        if self.dep_nets:
            self.dep_nets.create_joint_saver(self.name)

    def create_joint_saver(self, o_name):
        saver = {}
        name = o_name + '_' + self.name
        saver['out_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name)
        saver['train_data_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name + '_traindata')
        saver['ckpt_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name + '_ckpt')
        saver['saver'] = (tf.train.Saver(var_list=PoseTools.get_vars(self.net_name + '/'),
                                         max_to_keep=self.conf.maxckpt,
                                         save_relative_paths=True))
        self.saver = saver
        if self.dep_nets:
            self.dep_nets.create_joint_saver(name)


    def restore(self, sess, do_restore, at_step=-1):
        saver = self.saver
        name = self.net_name
        out_file = saver['out_file'].replace('\\', '/')
        latest_ckpt = tf.train.get_checkpoint_state(
            self.conf.cachedir, saver['ckpt_file'])
        if not latest_ckpt or not do_restore:
            start_at = 0
            sess.run(tf.variables_initializer(
                PoseTools.get_vars(name)),
                feed_dict=self.fd)
            print("Not loading {:s} variables. Initializing them".format(name))
        else:
            if at_step < 0:
                saver['saver'].restore(sess, latest_ckpt.model_checkpoint_path)
                match_obj = re.match(out_file + '-(\d*)', latest_ckpt.model_checkpoint_path)
                start_at = int(match_obj.group(1)) + 1
            else:
                aa = latest_ckpt.all_model_checkpoint_paths
                model_file = ''
                for a in aa:
                    match_obj = re.match(out_file + '-(\d*)', a)
                    step = int(match_obj.group(1))
                    if step >= at_step:
                        model_file = a
                        break
                saver['saver'].restore(sess, model_file)
                match_obj = re.match(out_file + '-(\d*)', model_file)
                start_at = int(match_obj.group(1)) + 1

        if self.dep_nets:
            self.dep_nets.restore_joint(sess, self.name, self.joint, do_restore)

        return start_at

    def restore_joint(self, sess, o_name, joint_train, do_restore):
        # when to restore and from to restore is kinda complicated.
        # if not doing joint training, then always restore from trained dependent model.
        # if doing joint training, but not restoring then again load from trained dependent model.
        # if doing joint training, and restoring but nothing exists then again load from trained dependent model.
        # in other words only load from jointly trained dependent model, if joint_train, do_restore and
        # an earlier saved model exists
        saver = self.saver
        name = o_name + '_' + self.name
        if joint_train and do_restore:
            latest_ckpt = tf.train.get_checkpoint_state(
                self.conf.cachedir, saver['ckpt_file'])
            if not latest_ckpt:
                ckpt_file = os.path.join(self.conf.cachedir,
                    self.conf.expname + '_' + self.name + '_ckpt')
                latest_ckpt = tf.train.get_checkpoint_state(
                    self.conf.cachedir, ckpt_file)
                assert latest_ckpt, 'dependent network {} hasnt been trained'.format(self.name)
        else:
            ckpt_file = os.path.join(self.conf.cachedir,
                                     self.conf.expname + '_' + self.name + '_ckpt')
            latest_ckpt = tf.train.get_checkpoint_state(
                self.conf.cachedir, ckpt_file)
            assert latest_ckpt, 'dependent network {} hasnt been trained'.format(self.name)

        saver['saver'].restore(sess, latest_ckpt.model_checkpoint_path)
        print("Loading {:s} variables from {:s}".format(name, latest_ckpt.model_checkpoint_path))

    def save(self, sess, step):
        saver = self.saver
        out_file = saver['out_file'].replace('\\', '/')
        saver['saver'].save(sess, out_file, global_step=step,
                            latest_filename=os.path.basename(saver['ckpt_file']))
        print('Saved state to %s-%d' % (out_file, step))
        if self.dep_nets and self.joint:
            self.dep_nets.save(sess, step)


    def init_td(self, td_fields):
        ex_td_fields = ['step']
        for t_f in td_fields:
            ex_td_fields.append('train_' + t_f)
            ex_td_fields.append('val_' + t_f)
        train_info = {}
        for t_f in ex_td_fields:
            train_info[t_f] = []
        self.train_info = train_info

    def restore_td(self):
        saver = self.saver
        train_data_file = saver['train_data_file'].replace('\\', '/')
        with open(train_data_file, 'rb') as td_file:
            if sys.version_info.major == 3:
                in_data = pickle.load(td_file, encoding='latin1')
            else:
                in_data = pickle.load(td_file)

            if not isinstance(in_data, dict):
                train_info, load_conf = in_data
                print('Parameters that do not match for {:s}:'.format(train_data_file))
                PoseTools.compare_conf(self.conf, load_conf)
            else:
                print("No config was stored for base. Not comparing conf")
                train_info = in_data
        self.train_info = train_info

    def save_td(self):
        saver = self.saver
        train_data_file = saver['train_data_file']
        with open(train_data_file, 'wb') as td_file:
            pickle.dump([self.train_info, self.conf], td_file, protocol=2)
        json_data = {}
        for x in self.train_info.keys():
            json_data[x] = np.array(self.train_info[x]).astype(np.float64).tolist()
        with open(train_data_file+'.json','w') as json_file:
            json.dump(json_data, json_file)

    def update_td(self, cur_dict):
        for k in cur_dict.keys():
            self.train_info[k].append(cur_dict[k])
        print_train_data(cur_dict)

    def create_ph(self):
        self.ph = {}
        learning_rate_ph = tf.placeholder(
            tf.float32, shape=[], name='learning_r')
        self.ph['learning_rate'] = learning_rate_ph
        self.ph['phase_train'] = tf.placeholder(
            tf.bool, name='phase_train')
        if self.dep_nets:
            self.dep_nets.create_ph()

    def create_fd(self):
        # to be subclassed
        return None

    def fd_train(self):
        return None

    def fd_val(self):
        return None

    def update_fd(self, db_type, sess, distort):
        return None

    def init_train(self, train_type):
        self.train_type = train_type
        self.create_ph()
        self.create_fd()
        self.open_dbs()

    def init_and_restore(self, sess, restore, td_fields, at_step=-1):
        self.create_cursors(sess)
        self.update_fd(self.DBType.Train, sess=sess, distort=True)
        start_at = self.restore(sess, restore, at_step)
        initialize_remaining_vars(sess)

        try:
            self.init_td(td_fields) if start_at is 0 else self.restore_td()
        except AttributeError: # If the conf file has been modified
            print('----------------')
            print("Couldn't load train data because the conf has changed!")
            print('----------------')
            self.init_td(td_fields)

        return start_at

    def train_step(self, step, sess, learning_rate):
        ex_count = step * self.conf.batch_size
        # cur_lr = learning_rate * self.conf.gamma ** math.floor(old_div(ex_count, self.conf.step_size))
        cur_lr = learning_rate * (self.conf.gamma ** (ex_count/ self.conf.step_size))
        self.fd[self.ph['learning_rate']] = cur_lr
        self.fd_train()
        self.update_fd(self.DBType.Train, sess, True)
        # doTrain = False
        # while not doTrain: # importance sampling
        #     self.update_fd(self.DBType.Train, sess, True)
        #     cur_loss = sess.run(self.cost,self.fd)
        #     x0 = np.median(self.train_loss)
        #     # x1 = np.percentile(self.train_loss,75)
        #     cur_tr = 1/(1+np.exp(-(cur_loss-x0)/x0))
        #     doTrain = (np.random.rand() < cur_tr)
        #     # cur_tr = np.percentile(self.train_loss,50)*np.random.rand()*2
        #     # if loss is approx 80 perc, then train half the time.
        #     # if loss > 2 * 80 perc, then always train.
        #     # doTrain = (cur_loss > cur_tr)
        #     self.train_loss[:-1] = self.train_loss[1:]
        #     self.train_loss[-1] = cur_loss

        sess.run(self.opt, self.fd)


    def setup_train(self, sess, distort=True, treat_as_val = False):
        if treat_as_val:
            self.fd_val()
        else:
            self.fd_train()
        self.update_fd(self.DBType.Train, sess=sess, distort=distort)

    def setup_val(self, sess):
        self.fd_val()
        self.update_fd(self.DBType.Val, sess=sess, distort=False)

    def create_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # plain vanilla.
            # self.opt = tf.train.AdamOptimizer(
            #     learning_rate=self.ph['learning_rate']).minimize(self.cost)

            # clipped gradients.
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.ph['learning_rate'])
            gradients, variables = zip(*optimizer.compute_gradients(self.cost))
            gradients = [ None if gradient is None else
                          tf.clip_by_norm(gradient, 5.0)
                    for gradient in gradients]
            self.opt = optimizer.apply_gradients(zip(gradients,variables))

    def compute_dist(self, preds, locs):
        tt1 = PoseTools.get_pred_locs(preds,self.edge_ignore) - locs
        tt1 = np.sqrt(np.sum(tt1 ** 2, 2))
        return np.nanmean(tt1)

    def compute_train_data(self, sess, db_type):
        self.setup_train(sess) if db_type is self.DBType.Train \
            else self.setup_val(sess)
        cur_loss, cur_pred = sess.run(
            [self.cost, self.pred], self.fd)
        cur_dist = self.compute_dist(cur_pred, self.locs)
        return cur_loss, cur_dist

    def train(self, restore, train_type, create_network,
              training_iters, loss, pred_in_key, learning_rate,
              td_fields=('loss', 'dist')):

        self.init_train(train_type)
        self.pred = create_network()
        self.cost = loss(self.pred, self.ph[pred_in_key])
        self.create_optimizer()
        self.create_saver()
        num_val_rep = self.conf.numTest / self.conf.batch_size + 1

        with tf.Session() as sess:
            start_at = self.init_and_restore(
                sess, restore, td_fields)

            if start_at < training_iters:
                for step in range(start_at, training_iters + 1):
                    self.train_step(step, sess, learning_rate)
                    if step % self.conf.display_step == 0:
                        train_loss, train_dist = self.compute_train_data(sess, self.DBType.Train)
                        val_loss = 0.
                        val_dist = 0.
                        for _ in range(num_val_rep):
                           cur_loss, cur_dist = self.compute_train_data(sess, self.DBType.Val)
                           val_loss += cur_loss
                           val_dist += cur_dist
                        val_loss = val_loss / num_val_rep
                        val_dist = val_dist / num_val_rep
                        cur_dict = {'step': step,
                                   'train_loss': train_loss, 'val_loss': val_loss,
                                   'train_dist': train_dist, 'val_dist': val_dist}
                        self.update_td(cur_dict)
                    if step % self.conf.save_step == 0:
                        self.save(sess, step)
                    if step % self.conf.save_td_step == 0:
                        self.save_td()
                print("Optimization Finished!")
                self.save(sess, training_iters)
                self.save_td()
            self.close_cursors()

    def classify_val(self,train_type=0, at_step=-1):

        if train_type is 0:
            val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        else:
            val_file = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename + '.tfrecords')
        num_val = 0
        for record in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        self.init_train(train_type)
        self.pred = self.create_network()
        saver = self.create_saver()

        with tf.Session() as sess:
            start_at = self.init_and_restore(sess, True, ['loss', 'dist'],at_step)

            val_dist = []
            val_ims = []
            val_preds = []
            val_predlocs = []
            val_locs = []
            for step in range(num_val/self.conf.batch_size):
                self.setup_val(sess)
                cur_pred = sess.run(self.pred, self.fd)
                cur_predlocs = PoseTools.get_pred_locs(cur_pred)
                cur_dist = np.sqrt(np.sum(
                    (cur_predlocs-self.locs) ** 2, 2))
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

        return val_dist, val_ims, val_preds, val_predlocs, val_locs

    def plot_results(self):
        saver = {}
        saver['train_data_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + self.name + '_traindata')
        self.saver = saver
        self.restore_td()
        n = 50
        plt.figure()
        plt.plot(moving_average(self.train_info['val_dist'],n),c='r')
        plt.plot(moving_average(self.train_info['train_dist'],n),c='g')

    def iter_res_image(self, max_iter, shape, train_type=0,
                       perc=[90,95,97,99],min_iter=0):
        ptiles = []
        for iter in range(min_iter,max_iter+1,self.conf.save_step):
            tf.reset_default_graph()
            A = self.classify_val(train_type,iter)
            ptiles.append(np.percentile(A[0],perc,axis=0))

        niter = len(ptiles)
        if A[1].shape[3] == 1:
            im = A[1][0,:,:,0]
        else:
            im = A[1][0,...]
        iszx = A[1].shape[2]
        iszy = A[1].shape[1]
        im = np.tile(im,shape)
        locs = A[4][0,...]
        alocs = []
        for yy in range(shape[1]):
            for xx in range(shape[0]):
                alocs.append(locs + [xx*iszx, yy*iszy])

        alocs = np.array(alocs).reshape([-1,2])
        nperc = ptiles[0].shape[0]
        ptiles_a = np.array(ptiles).transpose([1,0,2]).reshape([nperc,-1])
        PoseTools.create_result_image(im,alocs,ptiles_a)
        return ptiles


class PoseCommonMulti(PoseCommon):

    def create_cursors(self, sess):
        train_ims, train_locs, train_info = \
            multiResData.read_and_decode_multi(self.train_queue, self.conf)
        val_ims, val_locs, val_info = \
            multiResData.read_and_decode_multi(self.val_queue, self.conf)
        self.train_data = [train_ims, train_locs, train_info]
        self.val_data = [val_ims, val_locs, val_info]
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def compute_dist_m(self,preds,locs):
        pred_locs = PoseTools.get_pred_locs_multi(
            preds, self.conf.max_n_animals,
            self.conf.label_blur_rad * 2)

        dist = np.zeros(locs.shape[:-1])
        dist[:] = np.nan

        for ndx in range(self.conf.batch_size):
            for cls in range(self.conf.n_classes):
                for i_ndx in range(self.info[ndx][2][0]):
                    cur_locs = locs[ndx, i_ndx, cls, ...]
                    if np.isnan(cur_locs[0]):
                        continue
                    cur_dist = l2_dist(pred_locs[ndx, :, cls, :], cur_locs)
                    closest = np.argmin(cur_dist)
                    dist[ndx,i_ndx,cls] += cur_dist.min()

    def classify_val_m(self):
        val_file = os.path.join(self.conf.cachedir, self.conf.valfilename + '.tfrecords')
        num_val = 0
        for record in tf.python_io.tf_record_iterator(val_file):
            num_val += 1

        self.init_train(train_type=0)
        self.pred = self.create_network()
        saver = self.create_saver()

        with tf.Session() as sess:
            start_at = self.init_and_restore(sess, True, ['loss', 'dist'])

            val_dist = []
            val_ims = []
            val_preds = []
            val_locs = []
            for step in range(num_val/self.conf.batch_size):
                self.setup_val(sess)
                cur_pred = sess.run(self.pred, self.fd)
                cur_dist = self.compute_dist_m(cur_pred,self.locs)
                val_ims.append(self.xs)
                val_locs.append(self.locs)
                val_preds.append(cur_pred)
                val_dist.append(cur_dist)
            self.close_cursors()

        def val_reshape(in_a):
            in_a = np.array(in_a)
            return in_a.reshape( (-1,) + in_a.shape[2:])

        val_dist = val_reshape(val_dist)
        val_ims = val_reshape(val_ims)
        val_preds = val_reshape(val_preds)
        val_locs = val_reshape(val_locs)

        return val_dist, val_ims, val_preds, val_locs


class PoseCommonTime(PoseCommon):

    def create_cursors(self, sess):
        train_ims, train_locs, train_info = \
            multiResData.read_and_decode_time(self.train_queue, self.conf)
        val_ims, val_locs, val_info = \
            multiResData.read_and_decode_time(self.val_queue, self.conf)
        self.train_data = [train_ims, train_locs, train_info]
        self.val_data = [val_ims, val_locs, val_info]
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def read_images(self, db_type, distort, sess, shuffle=None):
        conf = self.conf
        cur_data = self.val_data if (db_type == self.DBType.Val)\
            else self.train_data
        xs = []
        locs = []
        info = []

        if shuffle is None:
            shuffle = distort

        # Tfrecords doesn't allow shuffling. Skipping a random
        # number of records
        # as a way to simulate shuffling. very hacky.

        for _ in range(conf.batch_size):
            if shuffle:
                for _ in range(np.random.randint(100)):
                    sess.run(cur_data)
            [cur_xs, cur_locs, cur_info] = sess.run(cur_data)
            xs.append(cur_xs)
            locs.append(cur_locs)
            info.append(cur_info)
        xs = np.array(xs)
        tw = (2*conf.time_window_size + 1)
        b_sz = conf.batch_size * tw
        xs = xs.reshape( (b_sz, ) + xs.shape[2:])
        locs = np.array(locs)
        locs = multiResData.sanitize_locs(locs)

        xs = PoseTools.adjust_contrast(xs, conf)

        # ideally normalize_mean should be here, but misc.imresize in scale_images
        # messes up dtypes. It converts float64 back to uint8.
        # so for now it'll be in update_fd.
        # xs = PoseTools.normalize_mean(xs, conf)
        if distort:
            if conf.horzFlip:
                xs, locs = PoseTools.randomly_flip_lr(xs, locs, tw)
            if conf.vertFlip:
                xs, locs = PoseTools.randomly_flip_ud(xs, locs, tw)
            xs, locs = PoseTools.randomly_rotate(xs, locs, conf, tw)
            xs, locs = PoseTools.randomly_translate(xs, locs, conf, tw)
            xs = PoseTools.randomly_adjust(xs, conf, tw)
        # else:
        #     rows, cols = xs.shape[2:]
        #     for ndx in range(xs.shape[0]):
        #         orig_im = copy.deepcopy(xs[ndx, ...])
        #         ii = copy.deepcopy(orig_im).transpose([1, 2, 0])
        #         mat = np.float32([[1, 0, 0], [0, 1, 0]])
        #         ii = cv2.warpAffine(ii, mat, (cols, rows))
        #         if ii.ndim == 2:
        #             ii = ii[..., np.newaxis]
        #         ii = ii.transpose([2, 0, 1])
        #         xs[ndx, ...] = ii

        self.xs = xs
        self.locs = locs
        self.info = info


class PoseCommonRNN(PoseCommonTime):

    def create_cursors(self, sess):
        train_ims, train_locs, train_info = \
            multiResData.read_and_decode_rnn(self.train_queue, self.conf)
        val_ims, val_locs, val_info = \
            multiResData.read_and_decode_rnn(self.val_queue, self.conf)
        self.train_data = [train_ims, train_locs, train_info]
        self.val_data = [val_ims, val_locs, val_info]
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def read_images(self, db_type, distort, sess, shuffle=None):
        conf = self.conf
        cur_data = self.val_data if (db_type == self.DBType.Val)\
            else self.train_data
        xs = []
        locs = []
        info = []

        if shuffle is None:
            shuffle = distort

        for _ in range(conf.batch_size):
            if shuffle:
                for _ in range(np.random.randint(100)):
                    sess.run(cur_data)
            [cur_xs, cur_locs, cur_info] = sess.run(cur_data)
            xs.append(cur_xs)
            locs.append(cur_locs)
            info.append(cur_info)
        xs = np.array(xs)
        tw = (conf.rnn_before + conf.rnn_after + 1)
        b_sz = conf.batch_size * tw
        xs = xs.reshape( (b_sz, ) + xs.shape[2:])
        locs = np.array(locs)
        locs = multiResData.sanitize_locs(locs)

        xs = PoseTools.adjust_contrast(xs, conf)

        # ideally normalize_mean should be here, but misc.imresize in scale_images
        # messes up dtypes. It converts float64 back to uint8.
        # so for now it'll be in update_fd.
        # xs = PoseTools.normalize_mean(xs, conf)
        if distort:
            if conf.horzFlip:
                xs, locs = PoseTools.randomly_flip_lr(xs, locs, tw)
            if conf.vertFlip:
                xs, locs = PoseTools.randomly_flip_ud(xs, locs, tw)
            xs, locs = PoseTools.randomly_rotate(xs, locs, conf, tw)
            xs, locs = PoseTools.randomly_translate(xs, locs, conf, tw)
            xs = PoseTools.randomly_adjust(xs, conf, tw)
        # else:
        #     rows, cols = xs.shape[2:]
        #     for ndx in range(xs.shape[0]):
        #         orig_im = copy.deepcopy(xs[ndx, ...])
        #         ii = copy.deepcopy(orig_im).transpose([1, 2, 0])
        #         mat = np.float32([[1, 0, 0], [0, 1, 0]])
        #         ii = cv2.warpAffine(ii, mat, (cols, rows))
        #         if ii.ndim == 2:
        #             ii = ii[..., np.newaxis]
        #         ii = ii.transpose([2, 0, 1])
        #         xs[ndx, ...] = ii

        self.xs = xs
        self.locs = locs
        self.info = info

