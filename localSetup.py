
# coding: utf-8

# In[2]:

import socket
import sys,re

if re.search('verman-ws1.janelia.priv',socket.gethostname()):
    sys.path.append('/groups/branson/home/kabram/bransonlab/pose_estimation/caffe/python')
    bdir = '/localhome/kabram/poseTF/'
elif re.search('bransonk-ws9.hhmi.org',socket.gethostname()):
    sys.path.append('/groups/branson/home/kabram/bransonlab/pose_estimation/caffe/python')
    bdir = '/groups/branson/bransonlab/mayank/PoseTF/'
elif re.search('bransonk-ws2',socket.gethostname()):
    sys.path.append('/groups/branson/home/kabram/bransonlab/pose_estimation/caffe/python')
    bdir = '/groups/branson/bransonlab/mayank/PoseTF/'
else:
    sys.path.append('/home/mayank/work/caffe/python')
    bdir = '/home/mayank/work/tensorflow/'


