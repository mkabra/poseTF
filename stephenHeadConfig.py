
# coding: utf-8

# In[ ]:

import os
import re
import localSetup

class myconfig(object): 

    # ----- Names

    expname = 'head'
    baseName = 'Base'
    fineName = 'Fine' #_resize'
    mrfName = 'MRF' #_identity'
    acName = 'AC'
    evalName = 'eval'

    # ----- Network parameters

    scale = 2
    rescale = 1  # how much to downsize the base image.
    numscale = 3
    pool_scale = 4
    # sel_sz determines the patch size used for the final decision
    # i.e., patch seen by the fc6 layer
    # ideally as large as possible but limited by
    # a) gpu memory size
    # b) overfitting due to large number of variables.
    sel_sz = 512/2/2/2
    psz = sel_sz/(scale**(numscale-1))/rescale/pool_scale
    dist2pos = 5
    label_blur_rad = 3 #1.5
    fine_label_blur_rad = 1.5
    n_classes = 5 # 
    dropout = 0.5 # Dropout, probability to keep units
    nfilt = 128
    nfcfilt = 512
    doBatchNorm = True
    useMRF = False
    useAC = True
    useHoldout = True
    device = None

    # ----- Fine Network parameters

    fine_flt_sz = 5
    fine_nfilt = 48
    fine_sz = 48

    # ----- MRF Network Parameters

    #maxDPts = 104*2
    #mrf_psz = (maxDPts/rescale)/pool_scale
    #Above should not be determined automatically
    # baseIter4MRFTrain = 4000 # without batch_norm
    baseIter4MRFTrain = 5000 # without batch_norm
    baseIter4ACTrain = 5000 # without batch_norm


    # ----- Learning parameters

    base_learning_rate = 0.0003 #0.0001 --without batch norm
    mrf_learning_rate = 0.00001
    ac_learning_rate = 0.0003
    fine_learning_rate = 0.0003

    batch_size = 8
    mult_fac = 16/batch_size
    base_training_iters = 5000*mult_fac
    # with rescale = 1 performance keeps improving even at around 3000 iters.. because batch size has been halved.. duh..
    # -- March 31, 2016 Mayank
    
    # with batch normalization quite good performance is achieved within 2000 iters
    # -- March 30, 2016 Mayank
    # when run iwth batch size of 32, best validation loss is achieved at 8000 iters 
    # for FlyHeadStephenCuratedData.mat -- Feb 11, 2016 Mayank
    fine_training_iters = 3000*mult_fac
    mrf_training_iters = 3000*mult_fac
    ac_training_iters = 5000*mult_fac
    eval_training_iters = 500*mult_fac
    gamma = 0.1
    step_size = 200000
    display_step = 30
    numTest = 100
    
    # range for contrast, brightness and rotation adjustment
    horzFlip = True
    vertFlip = False
    brange = [-0.2,0.2] 
    crange = [0.7,1.3]
    rrange = 30
    imax = 255.
    # fine_batch_size = 8


    # ----- Data parameters

    split = True
    view = 1  # view = 0 is side view view = 1 is front view
#    cropsz = 200 # cropping to be done based on imsz from now on.
    l1_cropsz = 0
#    imsz = (624,624) # This is after cropping. Orig is 1024x1024
    imsz = (512,512) # This is after cropping. Orig is 1024x1024
    map_size = 100000*psz**2*3
    cropLoc = {(1024,1024):[256,256],(512,768):[0,128]} # for front view crop the central part of the image

    cachedir = os.path.join(localSetup.bdir,'cacheHead/')
    labelfile = os.path.join(localSetup.bdir,'headTracking','FlyHeadStephenCuratedData_Janelia.mat')
 
#     cachedir = '/home/mayank/work/tensorflow/cacheHead/'
#     labelfile = '/home/mayank/work/tensorflow/headTracking/FlyHeadStephenCuratedData.mat'
#     labelfile = '/home/mayank/work/tensorflow/headTracking/FlyHeadStephenTestData_20160318.mat'
    viddir = '/groups/branson/bransonlab/mayank/PoseEstimationData/Stephen'
    ptn = 'fly_000[0-9]'
    trainfilename = 'train_lmdb'
    valfilename = 'val_lmdb'
    holdouttrain = 'holdouttrain_lmdb'
    holdouttest = 'holdouttest_lmdb'
    valdatafilename = 'valdata'
    valratio = 0.3
    holdoutratio = 0.8


    # ----- Save parameters

    save_step = 500
    maxckpt = 20
    baseoutname = expname + baseName
    fineoutname = expname + fineName
    mrfoutname = expname + mrfName
    acoutname = expname + acName
    evaloutname = expname + evalName
    baseckptname = baseoutname + 'ckpt'
    fineckptname = fineoutname + 'ckpt'
    mrfckptname = mrfoutname + 'ckpt'
    acckptname = acoutname + 'ckpt'
    evalckptname = evaloutname + 'ckpt'
    basedataname = baseoutname + 'traindata'
    finedataname = fineoutname + 'traindata'
    mrfdataname = mrfoutname + 'traindata'
    acdataname = acoutname + 'traindata'
    evaldataname = evaloutname + 'traindata'

    # ----- project specific functions

    def getexpname(self,dirname):
        dirname = os.path.normpath(dirname)
        dir_parts = dirname.split(os.sep)
        expname = dir_parts[-6] + "!" + dir_parts[-3] + "!" + dir_parts[-1][-10:-6]
        return expname

    def getexplist(self,L):
        fname = 'vid{:d}files'.format(self.view+1)
        return L[fname]

conf = myconfig()
sideconf = myconfig()
sideconf.cropLoc = {(1024,1024):[300,50],(512,768):[0,0]}
# sideconf.cachedir = '/home/mayank/work/tensorflow/cacheHeadSide/'
sideconf.cachedir = os.path.join(localSetup.bdir,'cacheHeadSide/')
sideconf.view = 0

