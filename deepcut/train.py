import logging
import threading

import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import load_config
from dataset.factory import create as create_dataset
from nnet.net_factory import pose_net
from nnet.pose_net import get_batch_spec
from util.mylogging import setup_logging

import os
import PoseTools
import numpy as np
import predict
from dataset.pose_dataset import Batch
import json

name = 'deeplabcut'

class LearningRate(object):
    def __init__(self, cfg):
#        self.steps = cfg.multi_step
        self.current_step = 0
        self.n_steps = cfg.dlc_steps
        self.gamma = cfg.gamma

    def get_lr(self, iteration):
        # lr = self.steps[self.current_step][0]
        # if iteration == self.steps[self.current_step][1]:
        #     self.current_step += 1
        lr = 0.0001 * (self.gamma ** (iteration*3/ self.n_steps))
        return lr


def setup_preloading(batch_spec):
    placeholders = {name: tf.placeholder(tf.float32, shape=spec) for (name, spec) in batch_spec.items()}
    names = placeholders.keys()
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20

    q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    enqueue_op = q.enqueue(placeholders_list)
    batch_list = q.dequeue()

    batch = {}
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders


def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
        batch_np = dataset.next_batch()
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        sess.run(enqueue_op, feed_dict=food)


def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.train.Coordinator()

    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t


def get_optimizer(loss_op, cfg):
    learning_rate = tf.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(cfg.adam_lr)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op

def save_td(cfg, train_info):
    train_data_file = os.path.join( cfg.cachedir, cfg.expname + '_' + name + '_traindata')
    json_data = {}
    for x in train_info.keys():
        json_data[x] = np.array(train_info[x]).astype(np.float64).tolist()
    with open(train_data_file + '.json', 'w') as json_file:
        json.dump(json_data, json_file)


def train(cfg):
    setup_logging()

#    cfg = load_config()
    dataset = create_dataset(cfg)
    train_info = {'train_dist':[],'train_loss':[],'val_dist':[],'val_loss':[],'step':[]}

    batch_spec = get_batch_spec(cfg)
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    net = pose_net(cfg)
    losses = net.train(batch)
    total_loss = losses['total_loss']
    outputs = [net.heads['part_pred'], net.heads['locref']]

    for k, t in losses.items():
        tf.summary.scalar(k, t)
    #merged_summaries = tf.summary.merge_all()

    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    restorer = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()

    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

#    train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

    learning_rate, train_op = get_optimizer(total_loss, cfg)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)

    #max_iter = int(cfg.multi_step[-1][1])
    max_iter = int(cfg.dlc_steps)
    display_iters = cfg.display_steps
    cum_loss = 0.0
    lr_gen = LearningRate(cfg)

    model_name = os.path.join( cfg.cachedir, cfg.expname + '_' + name)
    ckpt_file = os.path.join(cfg.cachedir, cfg.expname + '_' + name + '_ckpt')

    for it in range(max_iter+1):
        current_lr = lr_gen.get_lr(it)
        [_, loss_val] = sess.run([train_op, total_loss], # merged_summaries],
                                          feed_dict={learning_rate: current_lr})
        cum_loss += loss_val
 #       train_writer.add_summary(summary, it)

        if it % display_iters == 0:

            cur_out, batch_out = sess.run([outputs, batch], feed_dict={learning_rate: current_lr})
            scmap, locref = predict.extract_cnn_output(cur_out, cfg)

            # Extract maximum scoring location from the heatmap, assume 1 person
            loc_pred = predict.argmax_pose_predict(scmap, locref, cfg.stride)
            loc_in = batch_out[Batch.locs]
            dd = np.sqrt(np.sum(np.square(loc_pred[:,:,:2]-loc_in),axis=-1))

            average_loss = cum_loss / display_iters
            cum_loss = 0.0
            logging.info("iteration: {} loss: {} dist: {}  lr: {}"
                         .format(it, "{0:.4f}".format(average_loss),
                                 '{0:.2f}'.format(dd.mean()), current_lr))
            train_info['step'].append(it)
            train_info['train_loss'].append(loss_val)
            train_info['val_loss'].append(loss_val)
            train_info['val_dist'].append(dd.mean())
            train_info['train_dist'].append(dd.mean())

        if it % cfg.save_td_step == 0:
            save_td(cfg, train_info)
        # Save snapshot
        if (it % cfg.save_step == 0 and it != 0) or it == max_iter:
            saver.save(sess, model_name, global_step=it,
                       latest_filename=os.path.basename(ckpt_file))

    coord.request_stop()
    coord.join([thread])
    sess.close()


if __name__ == '__main__':
    train()
