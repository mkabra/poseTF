
# coding: utf-8

# In[2]:

import localSetup
import scipy.io as sio
import os,sys
import myutils
import re
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
from cvc import cvc
import math
import lmdb
import caffe
from random import randint,sample
import pickle
import h5py
import errno
import PoseTools


# In[ ]:

def findLocalDirs(conf):
    L = h5py.File(conf.labelfile,'r')
    localdirs = [u''.join(unichr(c) for c in L[jj[0]]) for jj in conf.getexplist(L)]
    seldirs = [True]*len(localdirs)
    L.close()
    return localdirs,seldirs


# In[4]:

def createValdata(conf,force=False):
    
    outfile = os.path.join(conf.cachedir,conf.valdatafilename)
    if ~force & os.path.isfile(outfile):
        return

    print('Creating val data %s!'%outfile)
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

def getMovieLists(conf):
    isval,localdirs,seldirs = loadValdata(conf)
    trainexps = []; valexps = []
    for ndx in range(len(localdirs)):
        if not seldirs[ndx]:
            continue
        if isval.count(ndx):
            valexps.append(localdirs[ndx])
        else:
            trainexps.append(localdirs[ndx])
    
    return trainexps, valexps


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

def createID(expname,curloc,fnum,imsz):
    for x in curloc: 
        assert x[0] >= 0,"x value %d is less than 0" %x[0]
        assert x[1] >= 0,"y value %d is less than 0" %x[1]
        assert x[0] < imsz[1],"x value %d is greater than imsz %d"%(x[0],imsz[1])
        assert x[1] < imsz[0],"y value %d is greater than imsz %d"%(x[1],imsz[0])
    
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

def createHoldoutData(conf):
    #Split the val data in hold out data for mrf training.
    isval,localdirs,seldirs = loadValdata(conf)
    n_ho = min(max(int(len(isval)*conf.holdoutratio),1),len(isval)-1)
#     ho_train = isval[0:n_ho]
#     ho_test = isval[(n_ho+1):]

    L = h5py.File(conf.labelfile,'r')
    pts = np.array(L['pts'])
    ts = np.array(L['ts']).squeeze().astype('int')
    expid = np.array(L['expidx']).squeeze().astype('int')
    view = conf.view
    traincount = 0; testcount = 0
    
    psz = conf.sel_sz
    map_size = 100000*conf.imsz[0]*conf.imsz[1]*8
    
    trainlmdbfilename =os.path.join(conf.cachedir,conf.holdouttrain)
    testlmdbfilename =os.path.join(conf.cachedir,conf.holdouttest)
    if os.path.isdir(trainlmdbfilename):
        shutil.rmtree(trainlmdbfilename)
    if os.path.isdir(testlmdbfilename):
        shutil.rmtree(testlmdbfilename)
    
    trainenv = lmdb.open(trainlmdbfilename, map_size=map_size)
    testenv = lmdb.open(testlmdbfilename, map_size=map_size)

    
    with trainenv.begin(write=True) as traintxn,testenv.begin(write=True) as testtxn:

        for ndx,dirname in enumerate(localdirs):

            if not seldirs[ndx]:
                continue
            if not isval.count(ndx):
                continue

            expname = conf.getexpname(dirname)
            frames = np.where(expid == (ndx + 1))[0]
            curdir = os.path.dirname(localdirs[ndx])
            cap = cv2.VideoCapture(localdirs[ndx])
            
            curtxn = testtxn if isval.index(ndx)>=n_ho else traintxn
                
            for curl in frames:

                fnum = ts[curl]
                if fnum > cap.get(cvc.FRAME_COUNT):
                    if fnum > cap.get(cvc.FRAME_COUNT)+1:
                        raise ValueError('Accessing frames beyond ' + 
                                         'the length of the video for' + 
                                         ' {} expid {:d} '.format(expname,ndx) + 
                                         ' at t {:d}'.format(fnum)
                                        )
                    continue
                framein = myutils.readframe(cap,fnum-1)
                cloc = conf.cropLoc[tuple(framein.shape[0:2])]
                framein = PoseTools.cropImages(framein,conf)
                framein = framein[:,:,0:1]
 
                curloc = np.round(pts[curl,:,view,:]).astype('int')
                curloc[:,0] = curloc[:,0] - cloc[1]  # ugh, the nasty x-y business.
                curloc[:,1] = curloc[:,1] - cloc[0]
                
                datum = createDatum(framein,1)
                str_id = createID(expname,curloc,fnum,conf.imsz)
                curtxn.put(str_id.encode('ascii'), datum.SerializeToString())

                if isval.index(ndx)>=n_ho:
                    testcount+=1
                else:
                    traincount+=1
                    
            cap.release() # close the movie handles
            print('Done %d of %d movies, train:%d test:%d' % (ndx,len(localdirs),traincount,testcount))
    trainenv.close() # close the database
    testenv.close()
    print('%d,%d number of pos examples added to the train db and test db' %(traincount,testcount))
    


# In[ ]:

def createDB(conf):

    L = h5py.File(conf.labelfile,'r')
    pts = np.array(L['pts'])
    ts = np.array(L['ts']).squeeze().astype('int')
    expid = np.array(L['expidx']).squeeze().astype('int')
    view = conf.view
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
                if fnum > cap.get(cvc.FRAME_COUNT):
                    if fnum > cap.get(cvc.FRAME_COUNT)+1:
                        raise ValueError('Accessing frames beyond ' + 
                                         'the length of the video for' + 
                                         ' {} expid {:d} '.format(expname,ndx) + 
                                         ' at t {:d}'.format(fnum)
                                        )
                    continue
                framein = myutils.readframe(cap,fnum-1)
                cloc = conf.cropLoc[tuple(framein.shape[0:2])]
                framein = PoseTools.cropImages(framein,conf)
                framein = framein[:,:,0:1]
 
                curloc = np.round(pts[curl,:,view,:]).astype('int')
                curloc[:,0] = curloc[:,0] - cloc[1]  # ugh, the nasty x-y business.
                curloc[:,1] = curloc[:,1] - cloc[0]
                
                datum = createDatum(framein,1)
                str_id = createID(expname,curloc,fnum,conf.imsz)
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
    
    

