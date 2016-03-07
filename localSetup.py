
# coding: utf-8

# In[2]:

import socket
import sys,re

if re.search('verman-ws1.janelia.priv',socket.gethostname()):
    sys.path.append('/groups/branson/home/kabram/bransonlab/pose_estimation/caffe/python')
    sys.path.append('/groups/branson/home/kabram/bransonlab/pyutils')
    bdir = '/localhome/kabram/poseTF/'
else:
    sys.path.append('/home/mayank/work/caffe/python')
    sys.path.append('/home/mayank/work/pyutils')
    bdir = '/home/mayank/work/tensorflow/'


