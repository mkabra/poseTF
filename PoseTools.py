
# coding: utf-8

# In[ ]:

import numpy as np
import scipy,re
import math,h5py
import caffe
from scipy import misc
from scipy import ndimage
import tensorflow as tf
import multiResData
import tempfile
import cv2
import PoseTrain
import localSetup
import myutils
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm


# In[ ]:

# not used anymore
# def scalepatches(patch,scale,num,rescale,cropsz):
#     sz = patch.shape
#     assert sz[0]%( (scale**(num-1))*rescale) is 0,"patch size isn't divisible by scale"
    
#     patches = []
#     patches.append(scipy.misc.imresize(patch[:,:,0],1.0/(scale**(num-1))/rescale))
#     curpatch = patch
#     for ndx in range(num-1):
#         sz = curpatch.shape
#         crop = int((1-1.0/scale)/2*sz[0])
# #         print(ndx,crop)
        
#         spatch = curpatch[crop:-crop,crop:-crop,:]
#         curpatch = spatch
#         sfactor = 1.0/(scale**(num-ndx-2))/rescale
#         tpatch = scipy.misc.imresize(spatch,sfactor,)
#         patches.append(tpatch[:,:,0])
#     return patches


# In[ ]:

def scaleImages(img,scale):
    sz = img.shape
    simg = np.zeros((sz[0],sz[1]/scale,sz[2]/scale,sz[3]))
    for ndx in range(sz[0]):
        for chn in range(sz[3]):
            simg[ndx,:,:,chn] = misc.imresize(img[ndx,:,:,chn],1./scale)
    return simg

def multiScaleImages(inImg, rescale, scale, l1_cropsz):
    # only crop the highest res image
    if l1_cropsz > 0:
        inImg_crop = inImg[:,l1_cropsz:-l1_cropsz,l1_cropsz:-l1_cropsz,:]
    else:
        inImg_crop = inImg
    x0_in = scaleImages(inImg_crop,rescale)
    x1_in = scaleImages(inImg,rescale*scale)
    x2_in = scaleImages(x1_in,scale)
    return x0_in,x1_in,x2_in


# In[ ]:

def processImage(framein,conf):
#     cropx = (framein.shape[0] - conf.imsz[0])/2
#     cropy = (framein.shape[1] - conf.imsz[1])/2
#     if cropx > 0:
#         framein = framein[cropx:-cropx,:,:]
#     if cropy > 0:
#         framein = framein[:,cropy:-cropy,:]
    framein = cropImages(framein,conf)
    framein = framein[np.newaxis,:,:,0:1]
    x0,x1,x2 = multiScaleImages(framein, conf.rescale,  conf.scale, conf.l1_cropsz)
    return x0,x1,x2
    


# In[ ]:

def cropImages(framein,conf):
    cshape = tuple(framein.shape[0:2])
    start = conf.cropLoc[cshape]  # cropLoc[0] should be for y.
    end = [conf.imsz[ndx] + start[ndx] for ndx in range(2)]
    return framein[start[0]:end[0],start[1]:end[1],:]


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

def readOnceLMDB(cursor,num,imsz,dataMod,reset=False):
    images = np.zeros((0,1,imsz[0],imsz[1]))
    locs = []
    datum = caffe.proto.caffe_pb2.Datum()
    
    if reset:
        cursor.first()
        
    done = False
    for ndx in range(num):
        if not cursor.next():
            done = True
            break
            
        value = cursor.value()
        key = cursor.key()
        datum.ParseFromString(value)
        expname,curloc,t = dataMod.decodeID(cursor.key())

        curlabel = datum.label
        data = caffe.io.datum_to_array(datum)
        images = np.append(images,data.squeeze()[np.newaxis,:,:],0)
        locs.append(curloc)
    return images,locs,done



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

def getBasePredLocs(pred,conf):
    predLocs = np.zeros([pred.shape[0],conf.n_classes,2])
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            maxndx = np.argmax(pred[ndx,:,:,cls])
            curloc = np.array(np.unravel_index(maxndx,pred.shape[1:3]))
            curloc = curloc * conf.pool_scale * conf.rescale
            predLocs[ndx,cls,0] = curloc[1]
            predLocs[ndx,cls,1] = curloc[0]
    return predLocs


# In[ ]:

def getFinePredLocs(pred,finepred,conf):
    predLocs = np.zeros([pred.shape[0],conf.n_classes,2])
    finepredLocs = np.zeros([pred.shape[0],conf.n_classes,2])
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            maxndx = np.argmax(pred[ndx,:,:,cls])
            curloc = np.array(np.unravel_index(maxndx,pred.shape[1:3]))
            curloc = curloc * conf.pool_scale * conf.rescale
            predLocs[ndx,cls,0] = curloc[1]
            predLocs[ndx,cls,1] = curloc[0]
            maxndx = np.argmax(finepred[ndx,:,:,cls])
            curfineloc = (np.array(np.unravel_index(maxndx,finepred.shape[1:3]))-conf.fine_sz/2)*conf.rescale
            finepredLocs[ndx,cls,0] = curloc[1] + curfineloc[1]
            finepredLocs[ndx,cls,1] = curloc[0] + curfineloc[0]
    return predLocs,finepredLocs


# In[ ]:

def getBaseError(locs,pred,conf):
    locerr = np.zeros(locs.shape)
    for ndx in range(pred.shape[0]):
        for cls in range(conf.n_classes):
            maxndx = np.argmax(pred[ndx,:,:,cls])
            predloc = np.array(np.unravel_index(maxndx,pred.shape[1:3]))
            predloc = predloc * conf.pool_scale * conf.rescale
            locerr[ndx][cls][0]= float(predloc[1])-locs[ndx][cls][0]
            locerr[ndx][cls][1]= float(predloc[0])-locs[ndx][cls][1]
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
                bfilt[psz/2+d12y,psz/2+d12x,c1,c2] += 1
    bfilt = (bfilt/pts.shape[0])
    return bfilt


# In[ ]:

def initMRFweightsIdentity(conf):
    L = h5py.File(conf.labelfile)
    pts = np.array(L['pts'])
    v = conf.view
    dx = np.zeros([pts.shape[0]])
    dy = np.zeros([pts.shape[0]])
    for ndx in range(pts.shape[0]):
        dx[ndx] = pts[ndx,:,v,0].max() - pts[ndx,:,v,0].min()
        dy[ndx] = pts[ndx,:,v,1].max() - pts[ndx,:,v,1].min()
    maxd = max(dx.max(),dy.max())
    hsz = int(math.ceil( (maxd*2/conf.rescale)/conf.pool_scale)/2)
    psz = hsz*2+1
#     psz = conf.mrf_psz
    bfilt = np.zeros([psz,psz,conf.n_classes,conf.n_classes])
    
    for c1 in range(conf.n_classes):
        bfilt[hsz,hsz,c1,c1] = 5.
    return bfilt


# In[ ]:

def getvars(vstr):
    varlist = tf.all_variables()
    blist = []
    for var in varlist:
        if re.match(vstr,var.name):
            blist.append(var)
    return blist


# In[ ]:

def compareConf(curconf,oldconf):
    ff = dir(curconf)
    for f in ff:
        if f[0:2] == '__' or f[0:3] == 'get':
            continue
        if getattr(curconf,f) != getattr(oldconf,f):
            print '%s doesnt match'%(f)


# In[ ]:

def createNetwork(conf,outtype):
    self = PoseTrain.PoseTrain(conf)
    self.createPH()
    self.createFeedDict()
    doBatchNorm = self.conf.doBatchNorm
    trainPhase = False
    self.feed_dict[self.ph['keep_prob']] = 1.
    tt = self.ph['y'].get_shape().as_list()
    tt[0] = 1
    self.feed_dict[self.ph['y']] = np.zeros(tt)
    tt = self.ph['locs'].get_shape().as_list()
    self.feed_dict[self.ph['locs']] = np.zeros(tt)
    
    
    with tf.variable_scope('base'):
        self.createBaseNetwork(doBatchNorm,trainPhase)
    self.createBaseSaver()

    if outtype > 1:
        with tf.variable_scope('mrf'):
            self.createMRFNetwork(doBatchNorm,trainPhase)
        self.createMRFSaver()

    if outtype > 2:
        with tf.variable_scope('fine'):
            self.createFineNetwork(doBatchNorm,trainPhase)
        self.createFineSaver()

    return self


# In[ ]:

def compareConf(curconf,oldconf):
    ff = dir(curconf)
    for f in ff:
        if f[0:2] == '__' or f[0:3] == 'get':
            continue
        if getattr(curconf,f) != getattr(oldconf,f):
            print '%s doesnt match'%(f)


# In[ ]:

def initNetwork(self,sess,outtype):
    self.restoreBase(sess,True)
    if outtype > 1:
        self.restoreMRF(sess,True)
    if outtype > 2:
        self.restoreFine(sess,True)
    self.initializeRemainingVars(sess)


# In[ ]:

def openMovie(moviename):
    cap = cv2.VideoCapture(moviename)
    nframes = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return cap,nframes


# In[ ]:

def createPredImage(predscores,n_classes):
    im = np.zeros(predscores.shape[0:2]+(3,))
    im[:,:,0] = np.argmax(predscores,2).astype('float32')/(n_classes)*180
    im[:,:,1] = (np.max(predscores,2)+1)/2*255
    im[:,:,2] = 255.
    im = np.clip(im,0,255)
    im = im.astype('uint8')
    return cv2.cvtColor(im,cv2.COLOR_HSV2RGB) 
        


# In[ ]:

def classifyMovie(conf,moviename,outtype,self,sess):
    cap,nframes = openMovie(moviename)
    predLocs = np.zeros([nframes,conf.n_classes,2,2])
    predmaxscores = np.zeros([nframes,conf.n_classes,2])
    
    if outtype == 3:
        if self.conf.useMRF:
            predPair = [self.mrfPred,self.finePred]
        else:
            predPair = [self.basePred,self.finePred]
    elif outtype == 2:
        predPair = [self.mrfPred,self.basePred]
    else:        
        predPair = [self.basePred]
        
    for curl in range(nframes):
        framein = myutils.readframe(cap,curl)
        x0,x1,x2 = processImage(framein,conf)
        self.feed_dict[self.ph['x0']] = x0
        self.feed_dict[self.ph['x1']] = x1
        self.feed_dict[self.ph['x2']] = x2
        pred = sess.run(predPair,self.feed_dict)
        if curl == 0:
            predscores = np.zeros((nframes,)+pred[0].shape[1:] + (2,))
        if outtype == 3:
            baseLocs,fineLocs = getFinePredLocs(pred[0],pred[1],conf)
            predLocs[curl,:,0,:] = fineLocs[0,:,:]
            predLocs[curl,:,1,:] = baseLocs[0,:,:]
            for ndx in range(conf.n_classes):
                predmaxscores[curl,:,0] = pred[0][0,:,:,ndx].max()
                predmaxscores[curl,:,1] = pred[1][0,:,:,ndx].max()
            predscores[curl,:,:,:,0] = pred[0]
        elif outtype == 2:
            baseLocs = getBasePredLocs(pred[0],conf)
            predLocs[curl,:,0,:] = baseLocs[0,:,:]
            baseLocs = getBasePredLocs(pred[1],conf)
            predLocs[curl,:,1,:] = baseLocs[0,:,:]
            for ndx in range(conf.n_classes):
                predmaxscores[curl,:,0] = pred[0][0,:,:,ndx].max()
                predmaxscores[curl,:,1] = pred[1][0,:,:,ndx].max()
            predscores[curl,:,:,:,0] = pred[0]
            predscores[curl,:,:,:,1] = pred[1]
        elif outtype == 1:
            baseLocs = getBasePredLocs(pred[0],conf)
            predLocs[curl,:,0,:] = baseLocs[0,:,:]
            for ndx in range(conf.n_classes):
                predmaxscores[curl,:,0] = pred[0][0,:,:,ndx].max()
            predscores[curl,:,:,:,0] = pred[0]

    cap.release()
    return predLocs,predscores,predmaxscores


# In[ ]:

def createPredMovie(conf,predList,moviename,outmovie,outtype):
    
    predLocs,predscores,predmaxscores = predList
#     assert false, 'stop here'
    tdir = tempfile.mkdtemp()

    cap = cv2.VideoCapture(moviename)
    nframes = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    
    cmap = cm.get_cmap('jet')
    rgba = cmap(np.linspace(0,1,conf.n_classes))
    
    fig = plt.figure(figsize = (9,4))
    for curl in range(nframes):
        framein = myutils.readframe(cap,curl)
        framein = cropImages(framein,conf)

        fig.clf()
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(framein[:,:,0], cmap=cm.gray)
        ax1.scatter(predLocs[curl,:,0,0],predLocs[curl,:,0,1], #hold=True,
                    c=cm.hsv(np.linspace(0,1-1./conf.n_classes,conf.n_classes)),
                    s=np.clip(predmaxscores[curl,:,0]*400,20,100),
                    linewidths=0,edgecolors='face')
        ax1.axis('off')
        ax2 = fig.add_subplot(1,2,2)
        rgbim = createPredImage(predscores[curl,:,:,:,0],conf.n_classes)
        ax2.imshow(rgbim)
        ax2.axis('off')

        fname = "test_{:06d}.png".format(curl)
        plt.savefig(os.path.join(tdir,fname))
        
    tfilestr = os.path.join(tdir,'test_*.png')
    mencoder_cmd = "mencoder mf://" + tfilestr +     " -frames " + "{:d}".format(nframes) + " -mf type=png:fps=15 -o " +     outmovie + " -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000000"
    os.system(mencoder_cmd)
    cap.release()
    return predLocs

