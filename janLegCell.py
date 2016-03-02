
# coding: utf-8

# In[2]:

import h5py
import numpy as np
L = h5py.File(conf.labelfile)
print(L['pts'])

# jj = np.array(L['pts'])


# In[3]:

import janLegConfig as conf
reload(conf)
import multiResData
reload(multiResData)

multiResData.createDB(conf)


# In[3]:

import os
import lmdb
import janLegConfig as conf

lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
env = lmdb.open(lmdbfilename, readonly = True)
print(env.stat())


# In[4]:

cursor = env.begin().cursor()


# In[6]:

import multiResData
import multiPawTools
import matplotlib.pyplot as plt
reload(multiResData)

ii,locs = multiPawTools.readLMDB(cursor,2,conf.imsz,multiResData)
locs = multiResData.sanitizelocs(locs)


# In[7]:

np.isnan(locs[1][0][0])


# In[1]:

plt.figure(figsize=(8,8))
import matplotlib.pyplot as plt
plt.gray()
plt.imshow(ii[0,0,:,:])

print(locs[0])

xlocs,ylocs  = zip(*locs[0])
plt.scatter(xlocs,ylocs,hold=True)


# In[1]:

import multiResTrain
reload(multiResTrain)
import janLegConfig as conf
reload(conf)

multiResTrain.trainBase(conf)


# In[ ]:

import multiResTrain
reload(multiResTrain)
import janLegConfig as conf
reload(conf)

multiResTrain.trainFine(conf)


# In[1]:

import scipy.io as sio
import os,sys
sys.path.append('/home/mayank/work/pyutils')
import myutils
import re
import janLegConfig as conf
import shutil

get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
import math
import lmdb
import caffe
from random import randint,sample
import pickle
import h5py

'''
Mayank Feb 3 2016
'''

import tensorflow as tf
import os,sys
import tempfile
import copy
sys.path.append('/home/mayank/work/caffe/python')
sys.path.append('/home/mayank/work/pyutils')

import caffe
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math
import cv2
import numpy as np
import scipy

import multiPawTools
import myutils
from convNetBase import *

# import stephenHeadConfig as conf
import multiResData

import mpld3
mpld3.enable_notebook()
from multiResTrain import *



learning_rate = conf.base_learning_rate;  
batch_size = conf.batch_size;        display_step = conf.display_step
n_input = conf.psz; n_classes = conf.n_classes; dropout = conf.dropout 
imsz = conf.imsz;   rescale = conf.rescale;     scale = conf.scale
pool_scale = conf.pool_scale

x0,x1,x2,y,keep_prob = createPlaceHolders(imsz,rescale,scale,pool_scale,n_classes)
locs_ph = tf.placeholder(tf.float32,[conf.batch_size,conf.n_classes,2])
print('Constructing Base Network')
weights = initNetConvWeights(conf)
pred_gradient,layers = net_multi_conv(x0,x1,x2, weights, keep_prob,
                      imsz,rescale,pool_scale)

print('Done Constructing Base Network')
baseoutname = '%s_%d.ckpt'%(conf.outname,conf.base_training_iters)
basemodelfile = os.path.join(conf.cachedir,baseoutname)


pred = pred_gradient
print('Constructing Fine Network')

# Construct fine model
labelT  = multiPawTools.createFineLabelTensor(conf)
layer1_1 = layers['base_dict_0']['conv1']
layer1_2 = layers['base_dict_0']['conv2']
layer2_1 = layers['base_dict_1']['conv1']
layer2_2 = layers['base_dict_1']['conv2']
curfine1_1 = extractPatches(layer1_1,pred,conf,1,4)
curfine1_2 = extractPatches(layer1_2,pred,conf,2,2)
curfine2_1 = extractPatches(layer2_1,pred,conf,2,2)
curfine2_2 = extractPatches(layer2_2,pred,conf,4,1)
curfine1_1u = tf.unpack(tf.transpose(curfine1_1,[1,0,2,3,4]))
curfine1_2u = tf.unpack(tf.transpose(curfine1_2,[1,0,2,3,4]))
curfine2_1u = tf.unpack(tf.transpose(curfine2_1,[1,0,2,3,4]))
curfine2_2u = tf.unpack(tf.transpose(curfine2_2,[1,0,2,3,4]))
finepred = fineOut(curfine1_1u,curfine1_2u,curfine2_1u,curfine2_2u,conf)    
limgs = multiPawTools.createFineLabelImages(locs_ph,pred,conf,labelT)
print('Done Constructing Fine Network')

# training data stuff
lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
env = lmdb.open(lmdbfilename, readonly = True)
valenv = lmdb.open(vallmdbfilename, readonly = True)

# Define loss and optimizer
costFine = tf.reduce_mean(tf.nn.l2_loss(finepred- tf.to_float(limgs)))
lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
env = lmdb.open(lmdbfilename, readonly = True)
valenv = lmdb.open(vallmdbfilename, readonly = True)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

fineoutname = '%s_%d.ckpt'%(conf.fineoutname,conf.fine_training_iters)
print('Restoring')
finemodelfile = os.path.join(conf.cachedir,fineoutname)
saver.restore(sess, finemodelfile)
print('Done Restoring from %s'%(fineoutname))

# init = tf.initialize_all_variables()
# sess.run(init)

valtxn = valenv.begin()
val_cursor = valtxn.cursor()

predError = np.zeros([0,conf.n_classes,2])
fineError = np.zeros([0,conf.n_classes,2])
for count in range(279/conf.batch_size):

    batch_xs, locs = multiPawTools.readLMDB(val_cursor,
                            conf.batch_size,imsz,multiResData)
    locs = multiResData.sanitizelocs(locs)

    x0_in,x1_in,x2_in = multiPawTools.multiScaleImages(
        batch_xs.transpose([0,2,3,1]),rescale,scale)

    labelims = multiPawTools.createLabelImages(locs,
                       conf.imsz,conf.pool_scale*conf.rescale,
                       conf.label_blur_rad) 
    feed_dict={x0: x0_in,x1: x1_in,x2: x2_in,
        y: labelims, keep_prob: dropout,locs_ph:np.array(locs)}
    out = sess.run([pred,finepred],feed_dict=feed_dict)
    curpred = out[0]
    curfinepred = out[1]
    curbaseError,curfineError = multiPawTools.getFineError(locs,curpred,curfinepred,conf)
    predError = np.concatenate([predError,curbaseError],0)
    fineError = np.concatenate([fineError,curfineError],0)

print('Done')

print(locs[-1])
print(out[0].shape)
maxndx = np.argmax(out[0][-1,:,:,0])
aa = np.array(np.unravel_index(maxndx,out[0].shape[1:3]))
print(maxndx,aa*8)
print(predError.mean())
print(fineError.mean())


# In[8]:

print(out[0].shape)
plt.imshow(out[0][1,:,:,0])
plt.show()
print(labelims.shape)
plt.imshow(labelims[1,:,:,0])
plt.show()
plt.imshow(batch_xs[1,0,:,:])
print(locs.shape)


# In[1]:

import h5py
import numpy as np
import stephenHeadConfig as conf
import PoseTools
reload(PoseTools)
import matplotlib.pyplot as plt
import tensorflow as tf

L = h5py.File(conf.labelfile)
pts = np.array(L['pts'])
ff = PoseTools.initMRFweights(conf)
ff.shape
plt.imshow(ff[:,:,0,1])
gg1 = tf.get_variable('dd',initializer=tf.constant(ff[:,:,0:0+1,1]))
gg2 = tf.get_variable('dd1',initializer=tf.constant(ff[:,:,1:1+1,1]))

kk = gg1+gg2+0
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
yy = kk.eval()


# In[2]:

yy.shape
plt.imshow(yy[:,:,0] )


# In[9]:

vvs = tf.trainable_variables()
for vv in vvs:
    print(vv.name,tf.Variable.get_shape(vv))

