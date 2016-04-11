
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


# In[2]:

from janLegConfig import conf as conf
import pickle

with open(conf.cachedir + '/' + conf.valdatafilename,'r') as f:
    isval,localdirs,seldirs = pickle.load(f)
    


# In[16]:

localdirs[0][35:]


# In[21]:

import re

newlocaldirs = []
for dd in localdirs:
    if re.search('Tracking_KAJ',dd):
        newlocaldirs.append('/home/mayank/work/PoseEstimationData/JanLegTracking/' + dd[34:])
    else:
        newlocaldirs.append('/home/mayank/work/PoseEstimationData/' + dd[36:])
with open(conf.cachedir + '/valdata','wb') as f:
    pickle.dump([isval,newlocaldirs,seldirs],f)


# In[1]:

from janLegConfig import conf as conf
import multiResData
multiResData.createHoldoutData(conf)


# In[ ]:

from janLegConfig import conf
import multiResData

