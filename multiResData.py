
# coding: utf-8

# In[2]:

import scipy.io as sio
import os,sys
sys.path.append('/home/mayank/work/pyutils')
import myutils
import re
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
import errno


# In[ ]:

def findLocalDirs(conf):
    L = h5py.File(conf.labelfile)
    localdirs = [u''.join(unichr(c) for c in L[jj[0]]) for jj in conf.getexplist(L)]
    seldirs = [True]*len(localdirs)
    L.close()
    return localdirs,seldirs


# In[4]:

def createValdata(conf,force=False):
    
    outfile = os.path.join(conf.cachedir,conf.valdatafilename)
    if ~force & os.path.isfile(outfile):
        return

    localdirs,seldirs = findLocalDirs(conf)
    nexps = len(seldirs)
    isval = sample(range(nexps),int(nexps*conf.valratio))
    try:
        os.makedirs(conf.cachedir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


    with open(outfile,'w') as f:
        pickle.dump([isval,localdirs,seldirs],f)


# In[6]:

def loadValdata(conf):
    
    outfile = os.path.join(conf.cachedir,conf.valdatafilename)
    assert os.path.isfile(outfile),"valdatafile doesn't exist"

    with open(outfile,'r') as f:
        isval,localdirs,seldirs = pickle.load(f)
    return isval,localdirs,seldirs


# In[ ]:

def createDatum(curp,label):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = curp.shape[0]
    datum.height = curp.shape[1]
    datum.width = curp.shape[2]
    datum.data = curp.tostring()  # or .tobytes() if numpy >= 1.9
    datum.label = label
    return datum


# In[ ]:

def createID(expname,curloc,fnum):
    xstr = '_'.join([str(x[0]) for x in curloc])
    ystr = '_'.join([str(x[1]) for x in curloc])
    
    str_id = '{:08d}:{}:x{}:y{}:t{:d}'.format(randint(0,1e8),
           expname,xstr,ystr,fnum)
    return str_id


# In[ ]:

def decodeID(keystr):
    vv = re.findall('(\d+):(.*):x(.*):y(.*):t(\d+)',keystr)[0]
    xlocs = [int(x) for x in vv[2].split('_')]
    ylocs = [int(x) for x in vv[3].split('_')]
    locs = zip(xlocs,ylocs)
    return vv[1],locs,int(vv[4])


# In[ ]:

def sanitizelocs(locs):
    nlocs = np.array(locs).astype('float')
    nlocs[nlocs<0] = np.nan
    return nlocs


# In[ ]:

def createDB(conf):

    L = h5py.File(conf.labelfile)
    pts = np.array(L['pts'])
    ts = np.array(L['ts']).squeeze().astype('int')
    expid = np.array(L['expidx']).squeeze().astype('int')
    view = conf.view
    cropsz = conf.cropsz
    count = 0; valcount = 0
    
    psz = conf.sel_sz
    map_size = 100000*conf.imsz[0]*conf.imsz[1]*8
    
    createValdata(conf)
    isval,localdirs,seldirs = loadValdata(conf)
    
    lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
    vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
    if os.path.isdir(lmdbfilename):
        shutil.rmtree(lmdbfilename)
    if os.path.isdir(vallmdbfilename):
        shutil.rmtree(vallmdbfilename)
    
    env = lmdb.open(lmdbfilename, map_size=map_size)
    valenv = lmdb.open(vallmdbfilename, map_size=map_size)

    
    with env.begin(write=True) as txn,valenv.begin(write=True) as valtxn:

        for ndx,dirname in enumerate(localdirs):
            if not seldirs[ndx]:
                continue

            expname = conf.getexpname(dirname)
            frames = np.where(expid == (ndx + 1))[0]
            curdir = os.path.dirname(localdirs[ndx])
            cap = cv2.VideoCapture(localdirs[ndx])
            
            curtxn = valtxn if isval.count(ndx) else txn
                
            for curl in frames:

                fnum = ts[curl]
                curloc = np.round(pts[curl,:,view,:]).astype('int')
                curloc = curloc - conf.cropsz
                if fnum > cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
                    if fnum > cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)+1:
                        raise ValueError('Accessing frames beyond ' + 
                                         'the length of the video for' + 
                                         ' {} expid {:d} '.format(expname,ndx) + 
                                         ' at t {:d}'.format(fnum)
                                        )
                    continue
                
                framein = myutils.readframe(cap,fnum-1)
                if cropsz > 0:
                    framein = framein[cropsz:-cropsz,cropsz:-cropsz,:]
                framein = framein[:,:,0:1]
                datum = createDatum(framein,1)
                str_id = createID(expname,curloc,fnum)
                curtxn.put(str_id.encode('ascii'), datum.SerializeToString())

                if isval.count(ndx):
                    valcount+=1
                else:
                    count+=1
                    
            cap.release() # close the movie handles
            print('Done %d of %d movies, count:%d val:%d' % (ndx,len(localdirs),count,valcount))
    env.close() # close the database
    valenv.close()
    print('%d,%d number of pos examples added to the db and valdb' %(count,valcount))
    
    

