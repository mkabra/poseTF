
# coding: utf-8

# In[1]:

# all the f'ing imports
import scipy.io as sio
import os,sys
sys.path.append('/home/mayank/work/pyutils')
import myutils
import re
from stephenHeadConfig import conf
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


import mpld3
mpld3.enable_notebook()

import multiResData
reload(multiResData)

multiResData.createDB(conf)


# In[1]:

import PoseTrain
reload(PoseTrain)
from stephenHeadConfig import conf

pobj = PoseTrain.PoseTrain(conf)
pobj.baseTrain(restore=True)


# In[ ]:

import PoseTrain
reload(PoseTrain)
import stephenHeadConfig as conf
import tensorflow as tf

pobj = PoseTrain.PoseTrain(conf)
pobj.mrfTrain(restore=False)


# In[ ]:

import PoseTrain
reload(PoseTrain)
import stephenHeadConfig as conf
import tensorflow as tf

pobj = PoseTrain.PoseTrain(conf)
pobj.fineTrain(restore=False)


# In[35]:

ll = np.max(berr,1)

zz = np.argwhere(ll>15)
from matplotlib import cm
fig = plt.figure(figsize=(8,5))
for idx in range(zz.shape[0]):
    img = zz[idx,0]

    fig.clf()
    ax12 = fig.add_subplot(1,1,1)
    ax12.imshow(all_images[img][0,:,:,0],cmap=cm.gray)

    all_locs = np.zeros([5,3,2])
    for ondx in range(conf.n_classes):
        maxndx = np.argmax(basepred[img][0,:,:,ondx])
        predloc = np.array(np.unravel_index(maxndx,basepred[img].shape[1:3]))
        predloc = predloc * conf.pool_scale 
        all_locs[ondx,0,:] = predloc
        maxndx = np.argmax(mrfpred[img][0,:,:,ondx])
        mrfloc = np.array(np.unravel_index(maxndx,mrfpred[img].shape[1:3]))
        mrfloc = mrfloc * conf.pool_scale 
        all_locs[ondx,1,:] = predloc
        maxndx = np.argmax(finepred[img][0,:,:,ondx])
        finepredloc = (np.array(np.unravel_index(maxndx,finepred[img].shape[1:3]))-conf.fine_sz/2)
        all_locs[ondx,2,:]= predloc+finepredloc


    plt.scatter(labels[img][0,:,0]/conf.rescale,labels[img][0,:,1]/conf.rescale,
                c=np.linspace(0,1,conf.n_classes),hold=True,cmap=cm.jet,
                linewidths=0,edgecolors='face',s=5)
    plt.scatter(all_locs[:,0,1],all_locs[:,0,0],
                c=np.linspace(0,1,conf.n_classes),hold=True,cmap=cm.jet,alpha=0.2,
                linewidths=0,edgecolors='face',s=5)
    plt.scatter(all_locs[:,2,1],all_locs[:,2,0],
                c=np.linspace(0,1,conf.n_classes),hold=True,cmap=cm.jet,alpha=0.6,
                linewidths=0,edgecolors='face',s=5)
    outname = 'results/headBaseAndFine%d.png'%idx
#     plt.savefig(outname,dpi=500)
#     raw_input('Press Enter')


# In[1]:

import localSetup
import PoseTools
reload(PoseTools)
import multiResData
reload(multiResData)
from stephenHeadConfig import conf
import os
import re

conf.batch_size = 1
conf.useMRF = False

_,valmovies = multiResData.getMovieLists(conf)
ndx = 3
mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))
PoseTools.createPredMovie(conf,valmovies[ndx],
                          '/home/mayank/work/tensorflow/results/headResults/movies/' + oname + '.avi',
                          1)


# In[1]:

import PoseTrain
reload(PoseTrain)
import PoseTools
reload (PoseTools)
import stephenHeadConfig as conf
conf.batch_size = 1
conf.useMRF = True

print('**** Setting batch size to %d! ****'%conf.batch_size)
print('**** USING MRF ****')
import tensorflow as tf

self = PoseTools.createNetwork(conf)
sess = tf.InteractiveSession()
PoseTools.initNetwork(self,sess)


# In[5]:

import multiResData
import os
import localSetup
import myutils
reload(PoseTools)

ndx = 0
curl = 70

_,valmovies = multiResData.getMovieLists(conf)
mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
cap,nframes = PoseTools.openMovie(valmovies[ndx])
framein = myutils.readframe(cap,curl)
x0,x1,x2 = PoseTools.processImage(framein,conf)
self.feed_dict[self.ph['x0']] = x0
self.feed_dict[self.ph['x1']] = x1
self.feed_dict[self.ph['x2']] = x2
pred = sess.run(self.finePred,self.feed_dict)
fig = plt.figure()

ax1 = fig.add_subplot(2,3,1)
ax1.imshow(pred[0,:,:,0],interpolation='nearest')
for ndx in range(4):
    ax = fig.add_subplot(2,3,ndx+2,sharex=ax1,sharey=ax1)
    ax.imshow(pred[0,:,:,ndx+1],interpolation='nearest')

