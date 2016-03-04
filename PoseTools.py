
# coding: utf-8

# In[ ]:

import numpy as np
import scipy,re
import math,h5py
import caffe
from scipy import misc
from scipy import ndimage
import tensorflow as tf

def scalepatches(patch,scale,num,rescale):
    sz = patch.shape
    assert sz[0]%( (scale**(num-1))*rescale) is 0,"patch size isn't divisible by scale"
    
    patches = []
    patches.append(scipy.misc.imresize(patch[:,:,0],1.0/(scale**(num-1))/rescale))
    curpatch = patch
    for ndx in range(num-1):
        sz = curpatch.shape
        crop = int((1-1.0/scale)/2*sz[0])
#         print(ndx,crop)
        
        spatch = curpatch[crop:-crop,crop:-crop,:]
        curpatch = spatch
        sfactor = 1.0/(scale**(num-ndx-2))/rescale
        tpatch = scipy.misc.imresize(spatch,sfactor,)
        patches.append(tpatch[:,:,0])
    return patches


# In[ ]:

def readLMDB(cursor,num,imsz,dataMod):
#     imsz = conf.imsz
    images = np.zeros((num,1,imsz[0],imsz[1]))
    locs = []
    datum = caffe.proto.caffe_pb2.Datum()

#     print(images.shape)
    for ndx in range(num):
        if not cursor.next():
            cursor.first()
#             print('restarting at %d' % ndx)
            
        value = cursor.value()
        key = cursor.key()
        datum.ParseFromString(value)
        expname,curloc,t = dataMod.decodeID(cursor.key())

        curlabel = datum.label
        data = caffe.io.datum_to_array(datum)
        images[ndx,0,:,:] = data.squeeze()
        locs.append(curloc)
    return images,locs



# In[ ]:

def blurLabel(imsz,loc,scale,blur_rad):
    sz0 = int(math.ceil(float(imsz[0])/scale))
    sz1 = int(math.ceil(float(imsz[1])/scale))

    label = np.zeros([sz0,sz1])
    if not np.isnan(loc[0]):
        label[int(loc[0]/scale),int(loc[1]/scale)] = 1
        blurL = ndimage.gaussian_filter(label,blur_rad)
        blurL = blurL/blurL.max()
    else:
        blurL = label
    return blurL


# In[ ]:

def scaleImages(img,scale):
    sz = img.shape
    simg = np.zeros((sz[0],sz[1]/scale,sz[2]/scale,sz[3]))
    for ndx in range(sz[0]):
        for chn in range(sz[3]):
            simg[ndx,:,:,chn] = misc.imresize(img[ndx,:,:,chn],1./scale)
    return simg

def multiScaleImages(inImg,rescale,scale):
    x0_in = scaleImages(inImg,rescale)
    x1_in = scaleImages(x0_in,scale)
    x2_in = scaleImages(x1_in,scale)
    return x0_in,x1_in,x2_in


# In[ ]:

def createLabelImages(locs,imsz,scale,blur_rad):
    n_classes = len(locs[0])
    sz0 = int(math.ceil(float(imsz[0])/scale))
    sz1 = int(math.ceil(float(imsz[1])/scale))

    labelims = np.zeros((len(locs),sz0,sz1,n_classes))
    for cls in range(n_classes):
        for ndx in range(len(locs)):
            modlocs = [locs[ndx][cls][1],locs[ndx][cls][0]]
            labelims[ndx,:,:,cls] = blurLabel(imsz,modlocs,scale,blur_rad)
 
    labelims = 2.0*(labelims-0.5)
    return labelims


# In[ ]:

def createFineLabelTensor(conf):
    tsz = int(conf.fine_sz + 2*6*math.ceil(conf.fine_label_blur_rad))
    timg = np.zeros((tsz,tsz))
    timg[tsz/2,tsz/2] = 1
    blurL = ndimage.gaussian_filter(timg,conf.fine_label_blur_rad)
    blurL = blurL/blurL.max()
    blurL = 2.0*(blurL-0.5)
    return tf.constant(blurL)


# In[ ]:

def extractFineLabelTensor(labelT,sz,dd,fsz):
    return tf.slice(labelT,dd+sz/2-fsz/2,[fsz,fsz])


# In[ ]:

def createFineLabelImages(locs,pred,conf,labelT):
    maxlocs = argmax2d(pred)*conf.pool_scale
    tsz = int(conf.fine_sz + 2*6*math.ceil(conf.fine_label_blur_rad))
    hsz = conf.fine_sz/2
    limgs = []
    for inum in range(conf.batch_size):
        curlimgs = []
        for ndx in range(conf.n_classes):
            dx = maxlocs[1,inum,ndx]-tf.to_int32(locs[inum,ndx,0]/conf.rescale)
            dy = maxlocs[0,inum,ndx]-tf.to_int32(locs[inum,ndx,1]/conf.rescale)
            dd = tf.pack([dx,dy])
            dd = tf.maximum(tf.to_int32(hsz-tsz/2),tf.minimum(tf.to_int32(tsz/2-hsz-1),dd))
            curlimgs.append(extractFineLabelTensor(labelT,tsz,dd,conf.fine_sz))
        limgs.append(tf.pack(curlimgs))
    return tf.transpose(tf.pack(limgs),[0,2,3,1])


# In[ ]:

def argmax2d(Xin):
    
    origShape = tf.shape(Xin)
    reshape_t = tf.concat(0,[origShape[0:1],[-1],origShape[3:4]])
    zz = tf.reshape(Xin,reshape_t)
    pp = tf.to_int32(tf.argmax(zz,1))
    sz1 = tf.slice(origShape,[2],[1])
    cc1 = tf.div(pp,tf.to_int32(sz1))
    cc2 = tf.mod(pp,tf.to_int32(sz1))
    
    return tf.pack([cc1,cc2])


# In[ ]:

def getBaseError(locs,pred,conf):
    locerr = np.zeros(locs.shape)
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            maxndx = np.argmax(pred[ndx,:,:,cls])
            predloc = np.unravel_index(maxndx,pred.shape[1:3])
            predloc = predloc * conf.pool_scale
            locerr[ndx][cls][0]= float(predloc[1])-loc[ndx][cls][0]
            locerr[ndx][cls][1]= float(predloc[0])-loc[ndx][cls][1]
    return locerr


# In[ ]:

def getFineError(locs,pred,finepred,conf):
    finelocerr = np.zeros([len(locs),conf.n_classes,2])
    baselocerr = np.zeros([len(locs),conf.n_classes,2])
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            maxndx = np.argmax(pred[ndx,:,:,cls])
            predloc = np.array(np.unravel_index(maxndx,pred.shape[1:3]))
            predloc = predloc * conf.pool_scale * conf.rescale
            maxndx = np.argmax(finepred[ndx,:,:,cls])
            finepredloc = (np.array(np.unravel_index(maxndx,finepred.shape[1:3]))-conf.fine_sz/2)*conf.rescale
            baselocerr[ndx,cls,0]= float(predloc[1])-locs[ndx][cls][0]
            baselocerr[ndx,cls,1]= float(predloc[0])-locs[ndx][cls][1]
            finelocerr[ndx,cls,0]= float(predloc[1]+finepredloc[1])-locs[ndx][cls][0]
            finelocerr[ndx,cls,1]= float(predloc[0]+finepredloc[0])-locs[ndx][cls][1]
    return baselocerr,finelocerr


# In[ ]:

def initMRFweights(conf):
    L = h5py.File(conf.labelfile)
    pts = np.array(L['pts'])
    v = conf.view
    dx = np.zeros([pts.shape[0]])
    dy = np.zeros([pts.shape[0]])
    for ndx in range(pts.shape[0]):
        dx[ndx] = pts[ndx,:,v,0].max() - pts[ndx,:,v,0].min()
        dy[ndx] = pts[ndx,:,v,1].max() - pts[ndx,:,v,1].min()
    maxd = max(dx.max(),dy.max())
    psz = int(math.ceil( (maxd*2/conf.rescale)/conf.pool_scale))
#     psz = conf.mrf_psz
    bfilt = np.zeros([psz,psz,conf.n_classes,conf.n_classes])
    
    for ndx in range(pts.shape[0]):
        for c1 in range(conf.n_classes):
            for c2 in range(conf.n_classes):
                d12x = pts[ndx,c1,v,0] - pts[ndx,c2,v,0]
                d12y = pts[ndx,c1,v,1] - pts[ndx,c2,v,1]
                d12x = int( (d12x/conf.rescale)/conf.pool_scale)
                d12y = int( (d12y/conf.rescale)/conf.pool_scale)
                bfilt[psz/2-d12y,psz/2-d12x,c1,c2] += 1
    bfilt = (bfilt/pts.shape[0])/pts.shape[1]
    return bfilt


# In[ ]:

def getvars(vstr):
    varlist = tf.all_variables()
    blist = []
    for var in varlist:
        if re.match(vstr,var.name):
            blist.append(var)
    return blist

