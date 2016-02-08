
# coding: utf-8

# In[ ]:

'''
Mayank Jan 12 2016
Paw detector modified from:
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf

import os,sys
sys.path.append('/home/mayank/work/caffe/python')

import caffe
import lmdb
import caffe.proto.caffe_pb2
# import pawconfig as conf

from caffe.io import datum_to_array
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import multiPawTools
import math

import cv2
import matplotlib.animation as manimation
sys.path.append('/home/mayank/work/pyutils')
import myutils
import tempfile
import copy


# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(
                l_input, w, strides=[1, 1, 1, 1], padding='SAME')
            ,b), 
        name=name)

def max_pool(name, l_input, k,s):
    return tf.nn.max_pool(
        l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], 
        padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(
        l_input, lsize, bias=1.0, alpha=0.0001 , beta=0.75, 
        name=name)

def upscale(name,l_input,sz):
    l_out = tf.image.resize_nearest_neighbor(l_input,sz,name=name)
    return l_out

def paw_net_multi_base(X,_weights):
    
    # X0
    conv1 = conv2d('conv1', X, _weights['wc1'], _weights['bc1'])
    pool1 = max_pool('pool1', conv1, k=3,s=2)
    norm1 = norm('norm1', pool1, lsize=2)
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _weights['bc2'])
    pool2 = max_pool('pool2', conv2, k=3,s=2)
    norm2 = norm('norm2', pool2, lsize=4)
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _weights['bc3'])
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _weights['bc4'])
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _weights['bc5'])
    return conv5

def initNetConvWeights(conf):
    # Store layers weight & bias
    nfilt = conf.nfilt
    nfcfilt = conf.nfcfilt
    n_classes = conf.n_classes
    rescale = conf.rescale
    pool_scale = conf.pool_scale
    
    sz5 = int(math.ceil(conf.psz/rescale/pool_scale))
    
    weights = {
        'base0': initBaseWeights(nfilt),
        'base1': initBaseWeights(nfilt),
        'base2': initBaseWeights(nfilt),       
        'wd1': tf.Variable(tf.random_normal([sz5,sz5,conf.numscale*nfilt,nfcfilt],stddev=0.005)),
        'wd2': tf.Variable(tf.random_normal([1,1,nfcfilt, nfcfilt],stddev=0.005)),
        'wd3': tf.Variable(tf.random_normal([1,1,nfcfilt, n_classes],stddev=0.01)),
        'bd1': tf.Variable(tf.ones([nfcfilt])),
        'bd2': tf.Variable(tf.ones([nfcfilt])),
        'bd3': tf.Variable(tf.zeros([n_classes]))
    }
    return weights

def initBaseWeights(nfilt):
    
    weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 48],stddev=0.01)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 48, nfilt],stddev=0.01)),
    'wc3': tf.Variable(tf.random_normal([3, 3, nfilt, nfilt],stddev=0.01)),
    'wc4': tf.Variable(tf.random_normal([3, 3, nfilt, nfilt],stddev=0.01)),
    'wc5': tf.Variable(tf.random_normal([3, 3, nfilt, nfilt],stddev=0.01)),
    'bc1': tf.Variable(tf.zeros([48])),
    'bc2': tf.Variable(tf.ones([nfilt])),
    'bc3': tf.Variable(tf.ones([nfilt])),
    'bc4': tf.Variable(tf.ones([nfilt])),
    'bc5': tf.Variable(tf.ones([nfilt]))
    }
    return weights

def net_multi_conv(X0,X1,X2, _weights,_dropout,imsz,rescale,pool_scale):
#     pool_scale = conf.pool_scale
    conv5_0 = paw_net_multi_base(X0,_weights['base0'])
    conv5_1 = paw_net_multi_base(X1,_weights['base1'])
    conv5_2 = paw_net_multi_base(X2,_weights['base2'])

    sz0 = int(math.ceil(float(imsz[0])/pool_scale/rescale))
    sz1 = int(math.ceil(float(imsz[1])/pool_scale/rescale))
    conv5_1_up = upscale('5_1',conv5_1,[sz0,sz1])
    conv5_2_up = upscale('5_2',conv5_2,[sz0,sz1])
    conv5_cat = tf.concat(3,[conv5_0,conv5_1_up,conv5_2_up])
    
    # Reshape conv5 output to fit dense layer input
    conv6 = conv2d('conv6',conv5_cat,_weights['wd1'],_weights['bd1']) 
    conv6 = tf.nn.dropout(conv6,_dropout)
    conv7 = conv2d('conv7',conv6,_weights['wd2'],_weights['bd2']) 
    conv7 = tf.nn.dropout(conv7,_dropout)
    # Output, class prediction
    out = tf.nn.bias_add(tf.nn.conv2d(
            conv7, _weights['wd3'], 
            strides=[1, 1, 1, 1], padding='SAME'),_weights['bd3'])
    return out


def createPlaceHolders(imsz,rescale,scale,pool_scale,n_classes):
#     imsz = conf.imsz
    # tf Graph input
    keep_prob = tf.placeholder(tf.float32) # dropout(keep probability)
    x0 = tf.placeholder(tf.float32, [None, 
                                     imsz[0]/rescale,
                                     imsz[1]/rescale,1])
    x1 = tf.placeholder(tf.float32, [None, 
                                     imsz[0]/scale/rescale,
                                     imsz[1]/scale/rescale,1])
    x2 = tf.placeholder(tf.float32, [None, 
                                     imsz[0]/scale/scale/rescale,
                                     imsz[1]/scale/scale/rescale,1])

    lsz0 = int(math.ceil(float(imsz[0])/pool_scale/rescale))
    lsz1 = int(math.ceil(float(imsz[1])/pool_scale/rescale))
    y = tf.placeholder(tf.float32, [None, lsz0,lsz1,n_classes])
    return x0,x1,x2,y,keep_prob

