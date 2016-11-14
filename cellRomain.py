
# coding: utf-8

# In[8]:

import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import bottomconf as conf
import multiResData
reload(multiResData)

multiResData.createTFRecordFromLbl(conf,split=False)


# In[3]:

import romainLegConfig
reload(romainLegConfig)
from romainLegConfig import side2conf as conf
import multiResData
reload(multiResData)

multiResData.createTFRecordFromLbl(conf,split=False)


# In[6]:

import PoseTrain
reload(PoseTrain)
from romainLegConfig import side2conf as conf
import tensorflow as tf

tf.reset_default_graph()
pobj = PoseTrain.PoseTrain(conf)
pobj.baseTrain(restore=False,trainType=1)


# In[1]:

import PoseTrain
reload(PoseTrain)
from romainLegConfig import side1conf as conf
import tensorflow as tf

tf.reset_default_graph()
pobj = PoseTrain.PoseTrain(conf)
pobj.mrfTrain(restore=True,trainType=1)


# In[4]:

import pickle
kk = pickle.load(open('/home/mayank/temp/temp.p','rb'))


# In[5]:

get_ipython().magic(u'pylab notebook')
print kk['tt'].shape
jj = kk['tt']
spt = 5
fig,ax = plt.subplots(4,5,figsize=(18,12))
ax = ax.flatten()
for ndx in range(jj.shape[3]):
    ax[ndx].imshow(kk['tt'][...,spt,ndx])


# In[18]:

import h5py
L = h5py.File(conf.labelfile)
print conf.labelfile
print L.keys()
pp = np.array(L['labeledpos'])
curpts = np.array(L[pp[0,1]])
print curpts.shape
frames = np.where(np.invert( np.isnan(curpts[:,0,0])))[0]
nptsPerView = np.array(L['cfg']['NumLabelPoints'])[0,0]
pts_st = int(conf.view*nptsPerView)
selpts = pts_st + conf.selpts
cc = curpts[:,:,selpts]
cc = cc[frames,:,:]
cc = cc.transpose([0,2,1])
print cc.shape
hh = np.zeros([0,18,2])
hh = np.append(hh,cc,axis=0)

print hh.shape


# In[1]:

import localSetup
import PoseTools
reload(PoseTools)
import multiResData
reload(multiResData)
import os
import re
import tensorflow as tf
from scipy import io
import cv2
from cvc import cvc

from romainLegConfig import side2conf as conf
conf.useMRF = False
outtype = 1
extrastr = ''
redo = True

tf.reset_default_graph()

self = PoseTools.createNetwork(conf,outtype)
sess = tf.InteractiveSession()
PoseTools.initNetwork(self,sess,outtype)


# In[2]:

valmovies = ['/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_1_date_2016_06_22_time_11_02_13_v001.avi']
for ndx in range(len(valmovies)):
    mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
    oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))
    pname = '/home/mayank/temp/romainOut/' + oname + extrastr
    if os.path.isfile(pname + '.mat') and not redo:
        continue
        
    predList = PoseTools.classifyMovie(conf,valmovies[ndx],outtype,self,sess,maxframes=1000)
#     PoseTools.createPredMovie(conf,predList,valmovies[ndx],pname + '.avi',outtype,maxframes=1000)


    cap = cv2.VideoCapture(valmovies[ndx])
    height = int(cap.get(cvc.FRAME_HEIGHT))
    width = int(cap.get(cvc.FRAME_WIDTH))
    orig_crop_loc = conf.cropLoc[(height,width)]
    crop_loc = [x/4 for x in orig_crop_loc] 
    end_pad = [height/4-crop_loc[0]-conf.imsz[0]/4,width/4-crop_loc[1]-conf.imsz[1]/4]
    pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0),(0,0)]
    predScores = np.pad(predList[1],pp,mode='constant',constant_values=-1.)

    predLocs = predList[0]
    predLocs[:,:,:,0] += orig_crop_loc[1]
    predLocs[:,:,:,1] += orig_crop_loc[0]
    
    io.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores[...,0],'expname':valmovies[ndx]})
    print 'Done:%s'%oname



# In[3]:

get_ipython().magic(u'debug')


# In[12]:

get_ipython().run_cell_magic(u'prun', u'-s cumulative', u"redo = True\nvalmovies = ['/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_2_date_2016_06_22_time_11_02_28_v001.avi']\nfor ndx in range(len(valmovies)):\n    mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))\n    oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))\n    pname = '/home/mayank/temp/romainOut/' + oname + extrastr\n    if os.path.isfile(pname + '.mat') and not redo:\n        continue\n        \n    predList = PoseTools.classifyMovie(conf,valmovies[ndx],outtype,self,sess,maxframes=1000)\n")


# In[18]:

import cv2
import cvc
cap = cv2.VideoCapture('/home/mayank/Dropbox/MultiViewFlyLegTracking/trackingJun22-11-02/bias_video_cam_2_date_2016_06_22_time_11_02_28_v001.avi')
s,i = cap.read()
fig, ax = plt.subplots()
ax.imshow(i)

gsize = 50
i_grey = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gsize,gsize))
ci = clahe.apply(i_grey)
fig, ax = plt.subplots()
ax.imshow(ci,cmap='gray')


# In[2]:

get_ipython().magic(u'pylab notebook')

import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/ipykernel/.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/widgets/.*')


# In[4]:

get_ipython().magic(u'pylab notebook')
import PoseTools
reload(PoseTools)
from romainLegConfig import side2conf as conf
oimg,dimg,ll = PoseTools.genDistortedImages(conf)
print oimg.shape


# In[5]:

for spt in range(8):
    fix,ax = plt.subplots(1,2)

    ax[0].imshow(oimg[spt,0,...],cmap='gray')
    ax[1].imshow(dimg[spt,...,0],cmap='gray')
    ax[1].scatter(ll[spt,:,0],ll[spt,:,1])

