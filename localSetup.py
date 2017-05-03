
# coding: utf-8

# In[2]:

import socket
import sys,re

if re.search('verman-ws1.hhmi.org',socket.gethostname()):
    sys.path.append('/groups/branson/home/kabram/bransonlab/pose_estimation/caffe/python')
    bdir = '/localhome/kabram/poseTF/'
elif re.search('bransonk-ws',socket.gethostname()):
    sys.path.append('/groups/branson/home/kabram/bransonlab/pose_estimation/caffe/python')
    bdir = '/groups/branson/bransonlab/mayank/PoseTF/'
else:
    sys.path.append('/home/mayank/work/caffe/python')
    bdir = '/home/mayank/work/poseEstimation/'


