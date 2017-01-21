
# coding: utf-8

# In[2]:




# In[1]:

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import math
import cv2
import copy
from cvc import cvc


# In[ ]:

def padgrab(inmat,padv,*args):
    assert len(args)%2==0,'number of coordinates should be even'
    assert (len(args)/2)==inmat.ndim,'number of coordinates should be even'
    cc = []
    padamt = []
    sz = inmat.shape
    idx = []
    for i in range(0,len(args),2):
        cc.append([args[i],args[i+1]])
        p1 = -min(args[i],0)
        p2 = max(args[i+1]-sz[i/2]+1,0)
        padamt.append([p1,p2])
        idx.append(slice(args[i]+p1,args[i+1]+p1))

    mmat = np.lib.pad(inmat,padamt,'constant',constant_values = padv)
    return mmat[idx]
    

    


# In[ ]:

# From: http://stackoverflow.com/questions/11469281/getting-individual-frames-using-cv-cap-prop-pos-frames-in-cvsetcaptureproperty
# because of course, videos have to be a pain in the a$$

def readframe(cap, position):
  assert position <= cap.get(cvc.FRAME_COUNT), "incorrect frame access" 
  positiontoset = position
  pos = -1
  cap.set(cvc.FRAME_POSITION, position)
  count =1
  while pos < position:
    pos = cap.get(cvc.FRAME_POSITION)
    ret, image = cap.read()
    if pos == position:
      return image
    elif pos > position:
      positiontoset -= 1
      cap.set(cvc.FRAME_POSITION, positiontoset)
      pos = -1
    count +=1
    
# def readframe(cap,fno):
#     cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fno)
#     dump,frame = cap.read()
#     assert dump is True, "Couldn't read the frame"
#     return frame

def nms(image,rad=3,thresh=0):
    image = copy.copy(image)
    roi = rad
    size = 2 * roi + 1
    image_max = ndimage.maximum_filter(image, size=size, mode='constant')
    mask = (image == image_max)
    imin = image.min()
    image[np.logical_not(mask)] = imin

    # Optionally find peaks above some threshold
    image[:roi,:] = imin
    image[-roi:,:] = imin
    image[:, :roi] = imin
    image[:, -roi:] = imin

    image_t = (image > thresh) * 1

    # get coordinates of peaks
    return np.transpose(image_t.nonzero())

