
# coding: utf-8

# In[ ]:

import numpy as np
import scipy
import math
import caffe
import pawData
import pawconfig as conf
from scipy import misc
from scipy import ndimage

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
        blurL = ndimage.gaussian_filter(label,blur_rad/scale)
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

