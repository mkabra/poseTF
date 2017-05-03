
# coding: utf-8

# In[1]:

# constant for different versions of opencv
import cv2
class cvc(object):
    (major,minor,_) = cv2.__version__.split('.')
    if major == '2':
        FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
        FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
        FRAME_POSITION = cv2.cv.CV_CAP_PROP_POS_FRAMES
    else:
        FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
        FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        FRAME_POSITION = cv2.CAP_PROP_POS_FRAMES
  
    

