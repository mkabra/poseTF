
# coding: utf-8

# In[ ]:

import os
import re
import localSetup
import numpy as np

class myconfig(object): 

    # ----- Names

    expname = 'romainLeg'
    baseName = 'Base'
    fineName = 'Fine' #_resize'
    mrfName = 'MRF' #_identity'
    acName = 'AC'
    regName = 'Reg'
    evalName = 'eval'
    genName = 'gen'

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
    n_classes = 18 # 
    dropout = 0.5 # Dropout, probability to keep units
    nfilt = 128
    nfcfilt = 512
    doBatchNorm = True
    useMRF = True
    useHoldout = False
    device = None
    reg_lambda = 0.5

    # ----- Fine Network parameters

    fine_flt_sz = 5
    fine_nfilt = 48
    fine_sz = 48

    # ----- MRF Network Parameters

    maxDPts = 400
    mrf_psz = (maxDPts/rescale)/pool_scale
    #Above should not be determined automatically
    # baseIter4MRFTrain = 4000 # without batch_norm
    baseIter4MRFTrain = 5000 # without batch_norm
    baseIter4ACTrain = 5000 # without batch_norm
    
    # ------ Pose Generation Network Params
    gen_minlen = 8


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
    basereg_training_iters = 5000*mult_fac
    fine_training_iters = 3000*mult_fac
    mrf_training_iters = 3000*mult_fac
    ac_training_iters = 5000*mult_fac
    eval_training_iters = 500*mult_fac
    gen_training_iters = 4000*mult_fac
    gamma = 0.1
    step_size = 200000
    display_step = 30
    numTest = 100
    
    # range for contrast, brightness and rotation adjustment
    horzFlip = False
    vertFlip = False
    brange = [0,0] #[-0.2,0.2] 
    crange = [1,1] #[0.7,1.3]
    rrange = 0
    imax = 255.
    adjustContrast = True
    clahegridsize = 20
    # fine_batch_size = 8


    # ----- Data parameters

    split = True
    view = 0  # view = 0 is side view view = 1 is front view
    l1_cropsz = 0
    imsz = (624,672) # This is after cropping. Orig is 1024x1024
    cropLoc = {(624,672):[0,0],(762,768):[85,0]} # for front view crop the central part of the image
    selpts = np.arange(0,18)

    cachedir = os.path.join(localSetup.bdir,'cache','romainLegBottom')
    labelfile = os.path.join(localSetup.bdir,'RomainLeg','Apr28AndJun22_onlyBottom.lbl')
 
    trainfilename = 'train_TF'
    fulltrainfilename = 'fullTrain_TF'
    valfilename = 'val_TF'
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
    evaloutname = expname + evalName
    genoutname = expname + genName
    baseregoutname = expname + regName
    baseckptname = baseoutname + 'ckpt'
    baseregckptname = baseregoutname + 'ckpt'
    fineckptname = fineoutname + 'ckpt'
    mrfckptname = mrfoutname + 'ckpt'
    evalckptname = evaloutname + 'ckpt'
    genckptname = genoutname + 'ckpt'
    basedataname = baseoutname + 'traindata'
    baseregdataname = baseregoutname + 'traindata'
    finedataname = fineoutname + 'traindata'
    mrfdataname = mrfoutname + 'traindata'
    evaldataname = evaloutname + 'traindata'
    gendataname = genoutname + 'traindata'

    # ----- project specific functions

    def getexpname(self,dirname):
        dirname = os.path.normpath(dirname)
        dir_parts = dirname.split(os.sep)
        expname = dir_parts[-1]
        return expname

    def getexplist(self,L):
        return L['movieFilesAll'][self.view,:]

bottomconf = myconfig()
side1conf = myconfig()
side1conf.cropLoc = {(592,288):[0,0]}
side1conf.view = 0  # view = 0 is side view view = 1 is front view
side1conf.imsz = (592,288) # This is after cropping. Orig is 1024x1024
side1conf.labelfile = os.path.join(localSetup.bdir,'RomainLeg','Jun22.lbl')
side1conf.selpts = np.arange(0,18)

side1conf.cachedir = os.path.join(localSetup.bdir,'cache','romainLegSide1')

