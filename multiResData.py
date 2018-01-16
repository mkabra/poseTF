from __future__ import print_function

# coding: utf-8

# In[2]:

from builtins import zip
from builtins import str
from builtins import chr
from builtins import range
import localSetup
import scipy.io as sio
import os,sys
import myutils
import re
import shutil

# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
from cvc import cvc
import math
import lmdb
# import caffe
from random import randint,sample
import pickle
import h5py
import errno
import PoseTools
import tensorflow as tf
import movies

def findLocalDirs(conf):
    L = h5py.File(conf.labelfile,'r')
    localdirs = [u''.join(chr(c) for c in L[jj]) for jj in conf.getexplist(L)]
    seldirs = [True]*len(localdirs)
    try:
        rdir = u''.join(chr(c) for c in L['projMacros']['rootdatadir'])
        ddir = u''.join(chr(c) for c in L['projMacros']['dataroot'])
        localdirs = [s.replace('$rootdatadir', rdir) for s in localdirs]
        localdirs = [s.replace('$dataroot', ddir) for s in localdirs]
    except:
        None
    L.close()
    return localdirs,seldirs

def get_trx_files(L,localdirs):
    trxfiles = [u''.join(chr(c) for c in L[jj]) for jj in L['trxFilesAll'][0]]
    movdir = [os.path.dirname(a) for a in localdirs]
    trxfiles = [s.replace('$movdir',m) for (s,m) in zip(trxfiles,movdir)]
    return trxfiles

def createValdata(conf,force=False):
    
    outfile = os.path.join(conf.cachedir,conf.valdatafilename)
    if ~force & os.path.isfile(outfile):
        return

    print('Creating val data %s!'%outfile)
    localdirs,seldirs = findLocalDirs(conf)
    nexps = len(seldirs)
    isval = []
    if (not hasattr(conf,'splitType')) or (conf.splitType is 'exp'):
        isval = sample(list(range(nexps)),int(nexps*conf.valratio))

    try:
        os.makedirs(conf.cachedir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


    with open(outfile,'w') as f:
        pickle.dump([isval,localdirs,seldirs],f)


def loadValdata(conf):
    
    outfile = os.path.join(conf.cachedir,conf.valdatafilename)
    assert os.path.isfile(outfile),"valdatafile {} doesn't exist".format(outfile)

    with open(outfile,'rb') as f:
        if sys.version_info.major == 3:
            isval,localdirs,seldirs = pickle.load(f,encoding='latin1')
        else:
            isval, localdirs, seldirs = pickle.load(f)
    return isval,localdirs,seldirs

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

def decodeID(keystr):
    vv = re.findall('(\d+):(.*):x(.*):y(.*):t(\d+)',keystr)[0]
    xlocs = [int(x) for x in vv[2].split('_')]
    ylocs = [int(x) for x in vv[3].split('_')]
    locs = list(zip(xlocs,ylocs))
    return vv[1],locs,int(vv[4])


def sanitizelocs(locs):
    nlocs = np.array(locs).astype('float')
    nlocs[nlocs<0] = np.nan
    return nlocs


# In[ ]:

# def createHoldoutData(conf):
#     #Split the val data in hold out data for mrf training.
#     isval,localdirs,seldirs = loadValdata(conf)
#     n_ho = min(max(int(len(isval)*conf.holdoutratio),1),len(isval)-1)
# #     ho_train = isval[0:n_ho]
# #     ho_test = isval[(n_ho+1):]

#     L = h5py.File(conf.labelfile,'r')
#     pts = np.array(L['pts'])
#     ts = np.array(L['ts']).squeeze().astype('int')
#     expid = np.array(L['expidx']).squeeze().astype('int')
#     view = conf.view
#     traincount = 0; testcount = 0
    
#     psz = conf.sel_sz
#     map_size = 100000*conf.imsz[0]*conf.imsz[1]*8
    
#     trainlmdbfilename =os.path.join(conf.cachedir,conf.holdouttrain)
#     testlmdbfilename =os.path.join(conf.cachedir,conf.holdouttest)
#     if os.path.isdir(trainlmdbfilename):
#         shutil.rmtree(trainlmdbfilename)
#     if os.path.isdir(testlmdbfilename):
#         shutil.rmtree(testlmdbfilename)
    
#     trainenv = lmdb.open(trainlmdbfilename, map_size=map_size)
#     testenv = lmdb.open(testlmdbfilename, map_size=map_size)

    
#     with trainenv.begin(write=True) as traintxn,testenv.begin(write=True) as testtxn:

#         for ndx,dirname in enumerate(localdirs):

#             if not seldirs[ndx]:
#                 continue
#             if not isval.count(ndx):
#                 continue

#             expname = conf.getexpname(dirname)
#             frames = np.where(expid == (ndx + 1))[0]
#             curdir = os.path.dirname(localdirs[ndx])
#             cap = cv2.VideoCapture(localdirs[ndx])
            
#             curtxn = testtxn if isval.index(ndx)>=n_ho else traintxn
                
#             for curl in frames:

#                 fnum = ts[curl]
#                 if fnum > cap.get(cvc.FRAME_COUNT):
#                     if fnum > cap.get(cvc.FRAME_COUNT)+1:
#                         raise ValueError('Accessing frames beyond ' + 
#                                          'the length of the video for' + 
#                                          ' {} expid {:d} '.format(expname,ndx) + 
#                                          ' at t {:d}'.format(fnum)
#                                         )
#                     continue
#                 framein = myutils.readframe(cap,fnum-1)
#                 cloc = conf.cropLoc[tuple(framein.shape[0:2])]
#                 framein = PoseTools.cropImages(framein,conf)
#                 framein = framein[:,:,0:1]
 
#                 curloc = np.round(pts[curl,:,view,:]).astype('int')
#                 curloc[:,0] = curloc[:,0] - cloc[1]  # ugh, the nasty x-y business.
#                 curloc[:,1] = curloc[:,1] - cloc[0]
                
#                 datum = createDatum(framein,1)
#                 str_id = createID(expname,curloc,fnum,conf.imsz)
#                 curtxn.put(str_id.encode('ascii'), datum.SerializeToString())

#                 if isval.index(ndx)>=n_ho:
#                     testcount+=1
#                 else:
#                     traincount+=1
                    
#             cap.release() # close the movie handles
#             print('Done %d of %d movies, train:%d test:%d' % (ndx,len(localdirs),traincount,testcount))
#     trainenv.close() # close the database
#     testenv.close()
#     print('%d,%d number of pos examples added to the train db and test db' %(traincount,testcount))
    


# In[ ]:

# def createDB(conf):

#     L = h5py.File(conf.labelfile,'r')
#     pts = np.array(L['pts'])
#     ts = np.array(L['ts']).squeeze().astype('int')
#     expid = np.array(L['expidx']).squeeze().astype('int')
#     view = conf.view
#     count = 0; valcount = 0
    
#     psz = conf.sel_sz
#     map_size = 100000*conf.imsz[0]*conf.imsz[1]*8
    
#     createValdata(conf)
#     isval,localdirs,seldirs = loadValdata(conf)
    
#     lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
#     vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
#     if os.path.isdir(lmdbfilename):
#         shutil.rmtree(lmdbfilename)
#     if os.path.isdir(vallmdbfilename):
#         shutil.rmtree(vallmdbfilename)
    
#     env = lmdb.open(lmdbfilename, map_size=map_size)
#     valenv = lmdb.open(vallmdbfilename, map_size=map_size)

    
#     with env.begin(write=True) as txn,valenv.begin(write=True) as valtxn:

#         for ndx,dirname in enumerate(localdirs):
#             if not seldirs[ndx]:
#                 continue

#             expname = conf.getexpname(dirname)
#             frames = np.where(expid == (ndx + 1))[0]
#             curdir = os.path.dirname(localdirs[ndx])
#             cap = cv2.VideoCapture(localdirs[ndx])
            
#             curtxn = valtxn if isval.count(ndx) else txn
                
#             for curl in frames:

#                 fnum = ts[curl]
#                 if fnum > cap.get(cvc.FRAME_COUNT):
#                     if fnum > cap.get(cvc.FRAME_COUNT)+1:
#                         raise ValueError('Accessing frames beyond ' + 
#                                          'the length of the video for' + 
#                                          ' {} expid {:d} '.format(expname,ndx) + 
#                                          ' at t {:d}'.format(fnum)
#                                         )
#                     continue
#                 framein = myutils.readframe(cap,fnum-1)
#                 cloc = conf.cropLoc[tuple(framein.shape[0:2])]
#                 framein = PoseTools.cropImages(framein,conf)
#                 framein = framein[:,:,0:1]
 
#                 curloc = np.round(pts[curl,:,view,:]).astype('int')
#                 curloc[:,0] = curloc[:,0] - cloc[1]  # ugh, the nasty x-y business.
#                 curloc[:,1] = curloc[:,1] - cloc[0]
                
#                 datum = createDatum(framein,1)
#                 str_id = createID(expname,curloc,fnum,conf.imsz)
#                 curtxn.put(str_id.encode('ascii'), datum.SerializeToString())

#                 if isval.count(ndx):
#                     valcount+=1
#                 else:
#                     count+=1
                    
#             cap.release() # close the movie handles
#             print('Done %d of %d movies, count:%d val:%d' % (ndx,len(localdirs),count,valcount))
#     env.close() # close the database
#     valenv.close()
#     print('%d,%d number of pos examples added to the db and valdb' %(count,valcount))
    
    


# In[ ]:

def _int64_feature(value):
    if not isinstance(value,(list,np.ndarray)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    if not isinstance(value,(list,np.ndarray)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# In[ ]:

def createTFRecord(conf):

    L = h5py.File(conf.labelfile,'r')
    pts = np.array(L['pts'])
    ts = np.array(L['ts']).squeeze().astype('int')
    expid = np.array(L['expidx']).squeeze().astype('int')
    view = conf.view
    count = 0; valcount = 0
    
    psz = conf.sel_sz
    
    createValdata(conf)
    isval,localdirs,seldirs = loadValdata(conf)
    
    trainfilename =os.path.join(conf.cachedir,conf.trainfilename)
    valfilename =os.path.join(conf.cachedir,conf.valfilename)
    
    env = tf.python_io.TFRecordWriter(trainfilename+'.tfrecords')
    valenv = tf.python_io.TFRecordWriter(valfilename+'.tfrecords')

    
    for ndx,dirname in enumerate(localdirs):
        if not seldirs[ndx]:
            continue

        expname = conf.getexpname(dirname)
        frames = np.where(expid == (ndx + 1))[0]
        curdir = os.path.dirname(localdirs[ndx])
        cap = cv2.VideoCapture(localdirs[ndx])

        curenv = valenv if isval.count(ndx) else env

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
            framein = PoseTools.crop_images(framein, conf)
            framein = framein[:,:,0:1]

            curloc = np.round(pts[curl,:,view,:]).astype('int')
            curloc[:,0] = curloc[:,0] - cloc[1]  # ugh, the nasty x-y business.
            curloc[:,1] = curloc[:,1] - cloc[0]
            curloc = curloc.clip(min=1)

            rows = framein.shape[0]
            cols = framein.shape[1]
            if np.ndim(framein) > 2:
                depth = framein.shape[2]
            else:
                depth = 1

            image_raw = framein.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'locs': _float_feature(curloc.flatten()),
                'expndx': _float_feature(ndx),
                'ts': _float_feature(curl),
                'image_raw': _bytes_feature(image_raw)}))
            curenv.write(example.SerializeToString())

            if isval.count(ndx):
                valcount+=1
            else:
                count+=1

        cap.release() # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx,len(localdirs),count,valcount))
    env.close() # close the database
    valenv.close()
    print('%d,%d number of pos examples added to the db and valdb' %(count,valcount))
    
    


# In[ ]:

def createFullTFRecord(conf):

    L = h5py.File(conf.labelfile,'r')
    pts = np.array(L['pts'])
    ts = np.array(L['ts']).squeeze().astype('int')
    expid = np.array(L['expidx']).squeeze().astype('int')
    view = conf.view
    count = 0; valcount = 0
    
    psz = conf.sel_sz
    
    createValdata(conf)
    isval,localdirs,seldirs = loadValdata(conf)
    
    trainfilename =os.path.join(conf.cachedir,conf.fulltrainfilename)
    
    env = tf.python_io.TFRecordWriter(trainfilename+'.tfrecords')
    
    for ndx,dirname in enumerate(localdirs):
        if not seldirs[ndx]:
            continue

        expname = conf.getexpname(dirname)
        frames = np.where(expid == (ndx + 1))[0]
        curdir = os.path.dirname(localdirs[ndx])
        cap = cv2.VideoCapture(localdirs[ndx])

        curenv = env

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
            framein = PoseTools.crop_images(framein, conf)
            framein = framein[:,:,0:1]

            curloc = np.round(pts[curl,:,view,:]).astype('int')
            curloc[:,0] = curloc[:,0] - cloc[1]  # ugh, the nasty x-y business.
            curloc[:,1] = curloc[:,1] - cloc[0]
            curloc = curloc.clip(min=0.1)

            rows = framein.shape[0]
            cols = framein.shape[1]
            if np.ndim(framein) > 2:
                depth = framein.shape[2]
            else:
                depth = 1

            image_raw = framein.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'locs': _float_feature(curloc.flatten()),
                'expndx': _float_feature(ndx),
                'ts': _float_feature(curl),
                'image_raw': _bytes_feature(image_raw)}))
            curenv.write(example.SerializeToString())

            count+=1

        cap.release() # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx,len(localdirs),count,valcount))
    env.close() # close the database
    print('%d,%d number of pos examples added to the db and valdb' %(count,valcount))
    
    


# In[ ]:

def createTFRecordFromLbl(conf,split=True):

    L = h5py.File(conf.labelfile,'r')
    
    
    psz = conf.sel_sz
    
    createValdata(conf)
    isval,localdirs,seldirs = loadValdata(conf)

    if split:
        trainfilename =os.path.join(conf.cachedir,conf.trainfilename)
        valfilename =os.path.join(conf.cachedir,conf.valfilename)

        env = tf.python_io.TFRecordWriter(trainfilename+'.tfrecords')
        valenv = tf.python_io.TFRecordWriter(valfilename+'.tfrecords')
    else:
        trainfilename =os.path.join(conf.cachedir,conf.fulltrainfilename)
        env = tf.python_io.TFRecordWriter(trainfilename+'.tfrecords')
        valenv = None
        

    pts = np.array(L['labeledpos'])
    
    view = conf.view
    count = 0; valcount = 0
    
    
    for ndx,dirname in enumerate(localdirs):
        if not seldirs[ndx]:
            continue

        expname = conf.getexpname(dirname)
        curpts = np.array(L[pts[0,ndx]])
        zz = curpts.reshape([curpts.shape[0],-1])
        frames = np.where(np.invert( np.all(np.isnan(curpts[:,:,:]),axis=(1,2))))[0]
        curdir = os.path.dirname(localdirs[ndx])
        cap = cv2.VideoCapture(localdirs[ndx])

        curenv = valenv if isval.count(ndx) and split else env
            

        for fnum in frames:

            if fnum > cap.get(cvc.FRAME_COUNT):
                if fnum > cap.get(cvc.FRAME_COUNT)+1:
                    raise ValueError('Accessing frames beyond ' + 
                                     'the length of the video for' + 
                                     ' {} expid {:d} '.format(expname,ndx) + 
                                     ' at t {:d}'.format(fnum)
                                    )
                continue
            framein = myutils.readframe(cap,fnum)
            cloc = conf.cropLoc[tuple(framein.shape[0:2])]
            framein = PoseTools.crop_images(framein, conf)
            framein = framein[:,:,0:1]

            nptsPerView = np.array(L['cfg']['NumLabelPoints'])[0,0]
            pts_st = int(view*nptsPerView)
            selpts = pts_st + conf.selpts
            curloc = curpts[fnum,:,selpts]
            curloc[:,0] = curloc[:,0] - cloc[1] - 1 # ugh, the nasty x-y business.
            curloc[:,1] = curloc[:,1] - cloc[0] - 1
            #-1 because matlab is 1-indexed
            curloc = curloc.clip(min=0,max=[conf.imsz[1]+7,conf.imsz[0]+7])
            

            rows = framein.shape[0]
            cols = framein.shape[1]
            if np.ndim(framein) > 2:
                depth = framein.shape[2]
            else:
                depth = 1

            image_raw = framein.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'locs': _float_feature(curloc.flatten()),
                'expndx': _float_feature(ndx),
                'ts': _float_feature(fnum),
                'image_raw': _bytes_feature(image_raw)}))
            curenv.write(example.SerializeToString())

            if isval.count(ndx) and split:
                valcount+=1
            else:
                count+=1

        cap.release() # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx,len(localdirs),count,valcount))
    env.close() # close the database
    if split:
        valenv.close()
    print('%d,%d number of pos examples added to the db and valdb' %(count,valcount))


def createTFRecordFromLblWithTrx(conf, split=True):

    createValdata(conf)
    is_val, local_dirs, sel_dirs = loadValdata(conf)

    L = h5py.File(conf.labelfile, 'r')
    npts_per_view = np.array(L['cfg']['NumLabelPoints'])[0, 0]
    pts = np.array(L['labeledpos'])
    trx_files = get_trx_files(L,local_dirs)
    d = L['projMacros']['dataroot']
    droot = u''.join(chr(c) for c in d)

    if split:
        trainfilename = os.path.join(conf.cachedir, conf.trainfilename)
        valfilename = os.path.join(conf.cachedir, conf.valfilename)

        env = tf.python_io.TFRecordWriter(trainfilename + '.tfrecords')
        valenv = tf.python_io.TFRecordWriter(valfilename + '.tfrecords')
    else:
        trainfilename = os.path.join(conf.cachedir, conf.fulltrainfilename)
        env = tf.python_io.TFRecordWriter(trainfilename + '.tfrecords')
        valenv = None

    view = conf.view
    count = 0
    valcount = 0

    for ndx, dirname in enumerate(local_dirs):
        if not sel_dirs[ndx]:
            continue

        expname = conf.getexpname(dirname)
        T = sio.loadmat(trx_files[ndx])['trx'][0]
        n_trx = len(T)
        curpts = np.array(L[pts[0, ndx]])-1
        # -1 for 1-indexing in matlab and 0-indexing in python
        trx_split = np.random.random(n_trx) < conf.valratio
        cap = movies.Movie(local_dirs[ndx])

        for trx_ndx in range(n_trx):
            frames = np.where(np.invert(np.all(np.isnan(curpts[trx_ndx,:, :, :]), axis=(1, 2))))[0]
            # cap = cv2.VideoCapture(localdirs[ndx])
            cur_trx = T[trx_ndx]

            for fnum in frames:

                if split:
                    if hasattr(conf,'splitType'):
                        if conf.splitType is 'frame':
                            curenv = valenv if np.random.random() < conf.valratio \
                                else env
                        elif conf.splitType is 'trx':
                            curenv = valenv if trx_split[trx_ndx] else env
                        else:
                            curenv = valenv if is_val.count(ndx) else env
                    else:
                        curenv = valenv if is_val.count(ndx) and split else env
                else:
                    curenv = env

                if fnum > cap.get_n_frames():#get(cvc.FRAME_COUNT):
                    if fnum > cap.get_n_frames: #get(cvc.FRAME_COUNT) + 1:
                        raise ValueError('Accessing frames beyond ' +
                                         'the length of the video for' +
                                         ' {} expid {:d} '.format(expname, ndx) +
                                         ' at t {:d}'.format(fnum)
                                         )
                    continue

                # framein = myutils.readframe(cap, fnum)
                framein = cap.get_frame(fnum)[0]
                if framein.ndim==2:
                    framein = framein[:,:,np.newaxis]
                trx_fnum = fnum - cur_trx['firstframe'][0,0]
                x = int(round(cur_trx['x'][0,trx_fnum]))-1
                y = int(round(cur_trx['y'][0,trx_fnum]))-1
                # -1 for 1-indexing in matlab and 0-indexing in python
                theta = cur_trx['theta'][0,trx_fnum]

                assert conf.imsz[0] == conf.imsz[1]
                framein, curloc = get_patch_trx(framein,x,y,theta,conf.imsz[0],
                                                curpts[trx_ndx,fnum,:,conf.selpts])
                framein = framein[:, :, 0:1]

                pts_st = int(view * npts_per_view)
                selpts = pts_st + conf.selpts

                rows = framein.shape[0]
                cols = framein.shape[1]
                if np.ndim(framein) > 2:
                    depth = framein.shape[2]
                else:
                    depth = 1

                image_raw = framein.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'locs': _float_feature(curloc.flatten()),
                    'expndx': _float_feature(ndx),
                    'ts': _float_feature(fnum),
                    'image_raw': _bytes_feature(image_raw)}))
                curenv.write(example.SerializeToString())

                if curenv is valenv:
                    valcount += 1
                else:
                    count += 1

        cap.close()  # close the movie handles
        print('Done %d of %d movies, count:%d val:%d' % (ndx, len(local_dirs), count, valcount))
    env.close()  # close the database
    if split:
        valenv.close()
    print('%d,%d number of pos examples added to the db and valdb' % (count, valcount))
    L.close()


def get_patch_trx(im,x,y,theta,psz, locs):
    im = im.copy()
    theta = theta + math.pi/2
    if im.ndim == 2:
        pad_im = np.pad(im,[psz,psz],'constant')
        patch = pad_im[y:y+2*psz,x:x+ 2*psz]
        rot_mat = cv2.getRotationMatrix2D((psz,psz),theta*180/math.pi,1)
        rpatch = cv2.warpAffine(patch, rot_mat, (2*psz, 2*psz) )
        rpatch = rpatch[psz/2:-psz/2,psz/2:-psz/2]
    else:
        pad_im = np.pad(im, [[psz, psz], [psz, psz], [0, 0]], 'constant')
        patch = pad_im[y:y + 2*psz, x:x + 2*psz,:]
        rot_mat = cv2.getRotationMatrix2D((psz,psz),theta*180/math.pi,1)
        rpatch = cv2.warpAffine(patch, rot_mat, (2*psz, 2*psz) )
        if rpatch.ndim == 2:
            rpatch = rpatch[:,:,np.newaxis]
        rpatch = rpatch[psz/2:-psz/2,psz/2:-psz/2,:]
    ll = locs.copy()
    ll = ll - [x,y]
    R = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    lr = np.dot(ll, R) + [psz/2, psz/2]
    return rpatch, lr


def read_and_decode(filename_queue,conf):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'height':tf.FixedLenFeature([], dtype=tf.int64),
          'width':tf.FixedLenFeature([], dtype=tf.int64),
          'depth':tf.FixedLenFeature([], dtype=tf.int64),
          'locs':tf.FixedLenFeature(shape=[conf.n_classes,2], dtype=tf.float32),
          'expndx': tf.FixedLenFeature([], dtype=tf.float32),
          'ts': tf.FixedLenFeature([], dtype=tf.float32),
          'image_raw':tf.FixedLenFeature([], dtype=tf.string)
                 })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'],tf.int64)
    width = tf.cast(features['width'],tf.int64)
    depth = tf.cast(features['depth'],tf.int64)
    if conf.imgDim > 1:
        image = tf.reshape(image,conf.imsz + (conf.imgDim,) )
    else:
        image = tf.reshape(image, conf.imsz)

    locs = tf.cast(features['locs'], tf.float64)
    expndx = tf.cast(features['expndx'],tf.float64)
    ts = tf.cast(features['ts'],tf.float64) #tf.constant([0]); #

    return image, locs, [expndx,ts]

def read_and_decode_multi(filename_queue,conf):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    n_max = conf.max_n_animals
    features = tf.parse_single_example(
        serialized_example,
        features={'height':tf.FixedLenFeature([], dtype=tf.int64),
          'width':tf.FixedLenFeature([], dtype=tf.int64),
          'depth':tf.FixedLenFeature([], dtype=tf.int64),
          'locs':tf.FixedLenFeature(shape=[n_max, conf.n_classes, 2], dtype=tf.float32),
          'n_animals': tf.FixedLenFeature(1, dtype=tf.int64),
          'expndx': tf.FixedLenFeature([], dtype=tf.float32),
          'ts': tf.FixedLenFeature([], dtype=tf.float32),
          'image_raw':tf.FixedLenFeature([], dtype=tf.string)
                 })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'],tf.int64)
    width = tf.cast(features['width'],tf.int64)
    depth = tf.cast(features['depth'],tf.int64)
    n_animals = tf.cast(features['n_animals'],tf.int64)
    if conf.imgDim > 1:
        image = tf.reshape(image,conf.imsz + (conf.imgDim,) )
    else:
        image = tf.reshape(image, conf.imsz)

    locs = tf.cast(features['locs'], tf.float64)
    expndx = tf.cast(features['expndx'],tf.float64)
    ts = tf.cast(features['ts'],tf.float64) #tf.constant([0]); #

    return image, locs, [expndx,ts,n_animals]

