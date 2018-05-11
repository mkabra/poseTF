
# coding: utf-8

# In[2]:

import socket
import sys,re

if re.search('verman-ws1.hhmi.org',socket.gethostname()):
    #sys.path.append('/groups/branson/home/kabram/bransonlab/pose_estimation/caffe/python')
#bdir = '/localhome/kabram/poseTF/'
    bdir = '/groups/branson/bransonlab/mayank/PoseTF/'
elif re.search('bransonk-ws',socket.gethostname()):
    bdir = '/groups/branson/bransonlab/mayank/PoseTF/'
elif re.search('int.janelia.org',socket.gethostname()):
    bdir = '/groups/branson/bransonlab/mayank/PoseTF/'
elif re.search('Keller-W12', socket.gethostname()):
    bdir = 'D:/mayank/data'
else:
    sys.path.append('/home/mayank/work/caffe/python')
    bdir = '/home/mayank/work/poseTF/'


