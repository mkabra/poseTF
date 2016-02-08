
# coding: utf-8

# In[2]:

import h5py
import numpy as np
L = h5py.File(conf.labelfile)
print(L['pts'])

# jj = np.array(L['pts'])


# In[1]:

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


# In[8]:

plt.figure(figsize=(8,8))
import matplotlib.pyplot as plt
plt.gray()
plt.imshow(ii[0,0,:,:])

print(locs[0])

xlocs,ylocs  = zip(*locs[0])
plt.scatter(xlocs,ylocs,hold=True)


# In[ ]:

import multiResTrain
reload(multiResTrain)
import janLegConfig as conf
reload(conf)

multiResTrain.train(conf)

