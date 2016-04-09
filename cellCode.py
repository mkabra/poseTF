
# coding: utf-8

# In[2]:

import pawData
reload(pawData)
import myutils
reload(myutils)
pawData.createDB()


# In[1]:

import sys
sys.path.append('/home/mayank/work/pyutils')
sys.path.append('/home/mayank/work/tensorflow')
import pawMulti
reload(pawMulti)
import myutils
reload(myutils)
import pawconfig
reload(pawconfig)
import multiPawTools
reload(multiPawTools)


pawMulti.train()


# In[75]:

sess = tf.InteractiveSession()


# In[6]:

import scipy.io as sio
import os,sys
sys.path.append('/home/mayank/work/pyutils')
import myutils
import re
import pawconfig as conf

get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
import math
import lmdb
import caffe
from random import randint
from multiPawTools import scalepatches


L = sio.loadmat(conf.labelfile)
pts = L['pts']
ts = L['ts']
expid = L['expidx']
    


# In[3]:

frames = np.where(expid[0,:]==4)[0]
fnum = ts[0,frames]
cap = cv2.VideoCapture('/home/mayank/Dropbox/AdamVideos/multiPoint/M118_20140730/M118_20140730_v002/movie_comb.avi')
print(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,1)
stat,framein1 = cap.read()
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,2)
stat,framein2 = cap.read()
framein1 = framein1.astype('float')
framein2 = framein2.astype('float')
ddff = framein1-framein2
ddff = np.abs(ddff).astype('uint8')
print(stat)
if stat:
    plt.imshow(ddff)

print(ddff.max())    
cap.release()


# In[4]:

import myutils
reload(myutils)
import cv2
cap = cv2.VideoCapture('/home/mayank/Dropbox/AdamVideos/multiPoint/M118_20140730/M118_20140730_v002/movie_comb.avi')
ff = myutils.readframe(cap,1998)
print(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT ))
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 1998)
print(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
stat,ff = cap.read()
print(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
print(stat)


# In[5]:

import moviepy.video.io.ffmpeg_reader as freader
reader = freader.FFMPEG_VideoReader('/home/mayank/Dropbox/AdamVideos/multiPoint/M118_20140730/M118_20140730_v002/movie_comb.avi')
f1 = reader.get_frame((-2.-0.1)/reader.fps)
f2 = reader.get_frame((1.-0.1)/reader.fps)
fe = reader.get_frame((1998.-0.1)/reader.fps)
type(f1)


# In[6]:

import cv2
cap = cv2.VideoCapture('/home/mayank/Dropbox/AdamVideos/movie_comb.avi')
          


# In[7]:

import pawconfig as conf
import scipy.io as sio
reload(conf)
L = sio.loadmat(conf.labelfile)
pts = L['pts']
ts = L['ts']
expid = L['expidx']
expid[0,3]


# In[1]:

import lmdb
env = lmdb.open('cacheHead/val_lmdb', readonly=True)
txn = env.begin()
print(env.stat())


# In[9]:

env.close()


# In[2]:

import caffe
import numpy as np
import re
import matplotlib.pyplot as plt
import pawData
reload(pawData)
cursor =txn.cursor()
cursor.first()


# In[6]:

import multiPawTools
reload(multiPawTools)
import pawconfig as conf
img,locs = multiPawTools.readLMDB(cursor,3,1)


# In[7]:

from scipy import misc
from scipy import ndimage
reload(multiPawTools)
reload(conf)
plt.gray()
ndx = 2
img = img.transpose([0,2,3,1])
plt.imshow(img[ndx,:,:,0])
plt.show()

blurL = multiPawTools.createLabelImages(locs,conf.imsz,conf.rescale*conf.pool_scale,
                                        conf.label_blur_rad,1)
x0 = multiPawTools.scaleImages(img,conf.rescale)
x1 = multiPawTools.scaleImages(x0,conf.scale)
x2 = multiPawTools.scaleImages(x1,conf.scale)
# labels = np.zeros([img.shape[2]/4,img.shape[3]/4])
# labels[int(locs[ndx][1])/4,int(locs[ndx][0])/4] = 1
# blurL = ndimage.gaussian_filter(labels,sigma = 3)
# blurL = blurL/blurL.max()
plt.imshow(blurL[ndx,:,:,0])
plt.show()
plt.imshow(x2[ndx,:,:,0])
plt.show()
print(blurL.max(),blurL.min())


# In[10]:

print(blurL[ndx,5:12,35:42,0])


# In[18]:

print(blurL.shape)


# In[55]:

from scipy import misc
sz = img.shape
scale =2
simg = np.zeros((sz[0],sz[1],sz[2]/scale,sz[3]/scale))
for ndx in range(sz[0]):
    for chn in range(sz[1]):
        simg[ndx,chn,:,:] = misc.imresize(img[ndx,chn,:,:],1./scale)
plt.gray()        
plt.imshow(simg[1,0,:,:])        
plt.show()
plt.imshow(img[1,0,:,:])        
plt.show()


# In[43]:

env.close()


# In[1]:

import pawMulti
reload(pawMulti)
import pawconfig as conf
reload(conf)
import tensorflow as tf

imsz = conf.imsz
x0 = tf.placeholder(tf.float32, [None, imsz[0],imsz[1],1])
x1 = tf.placeholder(tf.float32, [None, imsz[0]/2,imsz[1]/2,1])
x2 = tf.placeholder(tf.float32, [None, imsz[0]/4,imsz[1]/4,1])
dropout = tf.placeholder(tf.float32)
labelimg = tf.placeholder(tf.float32, [None, imsz[0]/4,imsz[1]/4,1])

weights = pawMulti.initNetConvWeights()
pred = pawMulti.paw_net_multi_conv(x0,x1,x2,weights,dropout)


# In[3]:

import numpy as np
imsz = conf.imsz
jj = np.ones([3,imsz[0],imsz[1],1])
jj1 = np.ones([3,imsz[0]/2,imsz[1]/2,1])
jj2 = np.ones([3,imsz[0]/4,imsz[1]/4,1])
sess.run(tf.initialize_all_variables())
out = sess.run(pred,feed_dict = {x0:jj,x1:jj1,x2:jj2,labelimg:jj2,dropout:1.})
print(out.shape)
print(jj2.shape)


# In[2]:

sess = tf.InteractiveSession()


# In[16]:

import tensorflow as tf
import numpy as np
x0 = tf.placeholder(tf.float32,[3,4])

jj = np.arange(12).reshape([2,3,2])
indices0 = tf.range(0,2*tf.shape(x0)[1],2)
indices1 = tf.range(1,2*tf.shape(x0)[1],2)
indices2 = tf.range(0,2*tf.shape(x0)[2],2)
indices3 = tf.range(1,2*tf.shape(x0)[2],2)

x1 = tf.transpose(tf.dynamic_stitch([indices0,indices1],[x0,x0]),[1,0])
x2 = tf.transpose(tf.dynamic_stitch([indices2,indices3],[x1,x1]),[1,0])

sess.run(tf.initialize_all_variables())
out = sess.run([x1,x2],feed_dict={x0:jj})
print(jj)
print(out[1])


# In[13]:

sess.close()


# In[ ]:

import tensorflow as tf

import os,sys
sys.path.append('/home/mayank/work/caffe/python')

import caffe
import lmdb
import caffe.proto.caffe_pb2
import pawconfig as conf

from caffe.io import datum_to_array
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import multiPawTools
import math
import pawMulti

learning_rate = conf.learning_rate
training_iters = conf.training_iters
batch_size = conf.batch_size
display_step = conf.display_step

# Network Parameters
n_input = conf.psz
n_classes = conf.n_classes # 
dropout = conf.dropout # Dropout, probability to keep units
imsz = conf.imsz
# tf Graph input
keep_prob = tf.placeholder(tf.float32) # dropout(keep probability)

x0 = tf.placeholder(tf.float32, [None, 
                                 imsz[0]/conf.rescale,
                                 imsz[1]/conf.rescale,1])
x1 = tf.placeholder(tf.float32, [None, 
                                 imsz[0]/conf.scale/conf.rescale,
                                 imsz[1]/conf.scale/conf.rescale,1])
x2 = tf.placeholder(tf.float32, [None, 
                                 imsz[0]/conf.scale/conf.scale/conf.rescale,
                                 imsz[1]/conf.scale/conf.scale/conf.rescale,1])

lsz0 = int(math.ceil(float(imsz[0])/conf.pool_scale/conf.rescale))
lsz1 = int(math.ceil(float(imsz[1])/conf.pool_scale/conf.rescale))
y = tf.placeholder(tf.float32, [None, lsz0,lsz1,n_classes])

lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
env = lmdb.open(lmdbfilename, map_size=conf.map_size)
valenv = lmdb.open(vallmdbfilename, map_size=conf.map_size)
txn = env.begin(write=True)
valtxn = valenv.begin(write=True)
train_cursor = txn.cursor()
val_cursor = valtxn.cursor()
weights = pawMulti.initNetConvWeights()

# Construct model
pred =pawMulti.paw_net_multi_conv(x0,x1,x2, weights, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.l2_loss(pred- y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

sess = tf.InteractiveSession()

sess.run(init)

saver.restore(sess, 'cache/pawMulti_r2_s3_20000.ckpt')
 

val_xs, locs = multiPawTools.readLMDB(val_cursor,batch_size*4,n_classes)
x0_in = multiPawTools.scaleImages(val_xs.transpose([0,2,3,1]),conf.rescale)
x1_in = multiPawTools.scaleImages(x0_in,conf.scale)
x2_in = multiPawTools.scaleImages(x1_in,conf.scale)
labelims = multiPawTools.createLabelImages(locs,
                           conf.imsz,conf.pool_scale*conf.rescale,
                           conf.label_blur_rad,1)
out = sess.run([pred,cost], feed_dict={x0:x0_in,
                                 x1:x1_in,
                                 x2:x2_in,
                           y: labelims, keep_prob: 1.})


# In[1]:

import matplotlib.pyplot as plt
from IPython import display
import time
plt.ion()
fig,axs = plt.subplots(1,3)
plt.gray()
for ndx in range(256):
    plt.sca(axs[0])
    plt.imshow(x0_in[ndx,:,:,0])
    display.clear_output(wait=True)
#     display.display(plt.gcf())
    plt.sca(axs[1])
    plt.imshow(out[0][ndx,:,:,0])
    display.clear_output(wait=True)
#     display.display(plt.gcf())
    plt.sca(axs[2])
    plt.imshow(labelims[ndx,:,:,0])
    display.clear_output(True)
    display.display(fig)
    time.sleep(1)


# In[28]:

import cv2
import matplotlib.animation as manimation
sys.path.append('/home/mayank/work/pyutils')
import myutils
import matplotlib
import tempfile


curdir = '/home/mayank/Dropbox/AdamVideos/multiPoint/M122_20140828/M122_20140828_v002'
tdir = tempfile.mkdtemp()
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# FFMpegWriter = manimation.writers['mencoder_file']
# writer = FFMpegWriter(fps=15,bitrate=2000)

fig = plt.figure()

cap = cv2.VideoCapture(os.path.join(curdir,'movie_comb.avi'))
nframes = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
plt.gray()
# with writer.saving(fig,"test_results.mp4",4):
count = 0
vidfilename = 'paw_detect.avi'
for fnum in range(nframes):
    plt.clf()
    framein = myutils.readframe(cap,fnum)
    framein = framein[np.newaxis,:,0:(framein.shape[1]/2),0:1]
    x0_in = multiPawTools.scaleImages(framein,conf.rescale)
    x1_in = multiPawTools.scaleImages(x0_in,conf.scale)
    x2_in = multiPawTools.scaleImages(x1_in,conf.scale)
    labelim = np.zeros([1,33,44,1])
    out = sess.run(pred, feed_dict={x0:x0_in,
                     x1:x1_in,
                     x2:x2_in,
                     y:labelim,
                     keep_prob: 1.})
    plt.imshow(x0_in[0,:,:,0])
    maxndx = np.argmax(out[0,:,:,0])
    loc = np.unravel_index(maxndx,out.shape[1:3])
    plt.scatter(loc[1]*4,loc[0]*4,hold=True)

    fname = "test_{:06d}.png".format(count)
    plt.savefig(os.path.join(tdir,fname))
    count+=1
#     plt.imshow(out[0,:,:,0])
#     fname = "test_heat_{:d}.png".format(fnum)
#     plt.savefig(fname)

#         writer.grab_frame()

ffmpeg_cmd = "ffmpeg -r 30 " + "-f image2 -i '/path/to/your/picName%d.png' -qscale 0 '/path/to/your/new/video.avi'

tfilestr = os.path.join(tdir,'test_*.png')
mencoder_cmd = "mencoder mf://" + tfilestr + " -frames " + "{:d}".format(count) + " -mf type=png:fps=15 -o " + vidfilename + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
print(mencoder_cmd)
os.system(mencoder_cmd)
cap.release()


# In[19]:

import pawData
a,b,c = pawData.loadValdata()


# In[1]:

import pawData
import pawMulti
import scipy.io as sio
import pawconfig as conf
reload(pawMulti)
reload(conf)
import os
import numpy as np
import tensorflow as tf
import tempfile
import matplotlib.pyplot as plt
import cv2
import sys,copy
sys.path.append('/home/mayank/work/pyutils')
import myutils


isval,localdirs,seldirs = pawData.loadValdata()
model_file = 'cache/pawMulti_r2_s3_20000.ckpt'
movcount = 0
maxcount = 5
L = sio.loadmat(conf.labelfile)
pts = L['pts']
ts = L['ts']
expid = L['expidx']

pred,saver,pholders = pawMulti.initPredSession()
tdir = tempfile.mkdtemp()

plt.gray()
# with writer.saving(fig,"test_results.mp4",4):
fig = plt.figure()

with tf.Session() as sess:
    saver.restore(sess, model_file)
    for ndx,dirname in enumerate(localdirs):
        if movcount> maxcount:
            break
        if not seldirs[ndx]:
            continue

        expname = os.path.basename(dirname)
        frames = np.where(expid[0,:] == (ndx + 1))[0]
        curdir = localdirs[ndx]
        outmovie = expname + ".avi"
        cap = cv2.VideoCapture(os.path.join(curdir,'movie_comb.avi'))
        nframes = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        count = 0
        for fnum in range(nframes):
            plt.clf()
            plt.axis('off')
            framein = myutils.readframe(cap,fnum)
            framein = framein[:,0:(framein.shape[1]/2),0:1            out = pawMulti.predict(copy.copy(framein),sess,pred,pholders)
            plt.imshow(framein[:,:,0])
            maxndx = np.argmax(out[0,:,:,0])
            loc = np.unravel_index(maxndx,out.shape[1:3])
            scalefactor = conf.rescale*conf.pool_scale
            plt.scatter(loc[1]*scalefactor,loc[0]*scalefactor,hold=True)

            fname = "test_{:06d}.png".format(count)
            plt.savefig(os.path.join(tdir,fname))
            count+=1

    #     ffmpeg_cmd = "ffmpeg -r 30 " + \
    #     "-f image2 -i '/path/to/your/picName%d.png' -qscale 0 '/path/to/your/new/video.avi'

        tfilestr = os.path.join(tdir,'test_*.png')
        mencoder_cmd = "mencoder mf://" + tfilestr + \
        " -frames " + "{:d}".format(count) + " -mf type=png:fps=15 -o " + \
        outmovie + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
    #     print(mencoder_cmd)
        os.system(mencoder_cmd)
        cap.release()

        movcount+=1


# In[1]:

import tensorflow as tf

sess = tf.InteractiveSession()

kk = tf.constant([3,-2,0.1,-0.05,5])
ss = tf.sign(kk)
mm = tf.mul(ss,tf.maximum(tf.abs(kk)-0.2,0))
aa = mm.eval()
print(aa)


# In[23]:

import lmdb
lmdbfilename= 'cacheHeadSide/train_lmdb'
env = lmdb.open(lmdbfilename, readonly = True)


txn = env.begin()
print(txn.stat()['entries'])


# In[24]:

import PoseTools
import multiResData
cursor = txn.cursor()
ii,ll = PoseTools.readLMDB(cursor,1,[512, 512],multiResData)
print ii.shape
print ll
plt.imshow(ii[0,0,:,:])


# In[11]:

import pickle

with open('cacheHead/headMRFtraindata','rb') as f:
    gg = pickle.load(f)
    


# In[37]:

print gg[0].keys()
plt.clf()
x = gg[0]['step_no'][5:]
plt.plot(x,gg[0]['val_base_dist'][5:])
plt.plot(x,gg[0]['val_dist'][5:], hold=True)
plt.legend(('base','mrf'))


# In[40]:

from stephenHeadConfig import conf as conf
print conf.cachedir

