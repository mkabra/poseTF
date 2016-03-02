
# coding: utf-8

# In[ ]:

'''
Mayank Feb 8 2016
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


