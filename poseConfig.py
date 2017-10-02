from __future__ import division

# coding: utf-8

# In[ ]:

from builtins import object
from past.utils import old_div
import os
import re
import localSetup
import numpy as np


class myconfig(object):
    # ----- Names

    expname = 'head'
    baseName = 'Base'
    fineName = 'Fine'  # _resize'
    mrfName = 'MRF'  # _identity'
    acName = 'AC'
    regName = 'Reg'
    evalName = 'eval'
    genName = 'gen'

    # ----- Network parameters

    scale = 2
    rescale = 1  # how much to downsize the base image.
    numscale = 3
    pool_scale = 4
    pool_size = 3
    pool_stride = 2
    # sel_sz determines the patch size used for the final decision
    # i.e., patch seen by the fc6 layer
    # ideally as large as possible but limited by
    # a) gpu memory size
    # b) overfitting due to large number of variables.
    sel_sz = 512 / 2 / 2
    psz = sel_sz / (scale ** (numscale - 1)) / rescale / pool_scale
    dist2pos = 5
    label_blur_rad = 3  # 1.5
    fine_label_blur_rad = 1.5
    n_classes = 5  #
    # dropout = 0.5 # Dropout, probability to keep units
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

    # maxDPts = 104*2
    # mrf_psz = (maxDPts/rescale)/pool_scale
    # Above should not be determined automatically
    # baseIter4MRFTrain = 4000 # without batch_norm
    baseIter4MRFTrain = 5000  # without batch_norm
    baseIter4ACTrain = 5000  # without batch_norm

    # ------ Pose Generation Network Params
    gen_minlen = 8

    # ----- Learning parameters

    base_learning_rate = 0.0003  # 0.0001 --without batch norm
    mrf_learning_rate = 0.00001
    ac_learning_rate = 0.0003
    fine_learning_rate = 0.0003

    batch_size = 8
    mult_fac = old_div(16, batch_size)
    base_training_iters = 10000 * mult_fac  # 15000
    # with rescale = 1 performance keeps improving even at around 3000 iters.. because batch size has been halved.. duh..
    # -- March 31, 2016 Mayank

    # with batch normalization quite good performance is achieved within 2000 iters
    # -- March 30, 2016 Mayank
    # when run iwth batch size of 32, best validation loss is achieved at 8000 iters
    # for FlyHeadStephenCuratedData.mat -- Feb 11, 2016 Mayank
    basereg_training_iters = 5000 * mult_fac
    fine_training_iters = 5000 * mult_fac
    mrf_training_iters = 3000 * mult_fac
    ac_training_iters = 5000 * mult_fac
    eval_training_iters = 500 * mult_fac
    gen_training_iters = 4000 * mult_fac
    gamma = 0.1
    step_size = 100000
    display_step = 30
    numTest = 100

    # range for contrast, brightness and rotation adjustment
    horzFlip = False
    vertFlip = False
    brange = [-0.2, 0.2]
    crange = [0.7, 1.3]
    rrange = 30
    trange = 0
    imax = 255.
    adjustContrast = False
    clahegridsize = 20
    normalize_mean_img = True

    # fine_batch_size = 8


    # ----- Data parameters
    l1_cropsz = 0
    split = True
    trainfilename = 'train_TF'
    fulltrainfilename = 'fullTrain_TF'
    valfilename = 'val_TF'
    holdouttrain = 'holdouttrain_TF'
    holdouttest = 'holdouttest_TF'
    valdatafilename = 'valdata'
    valratio = 0.3
    holdoutratio = 0.8

    # ----- MDN params
    mdn_min_sigma = 3.
    mdn_max_sigma = 4.

    # ----- Save parameters

    save_step = 500
    maxckpt = 20
    # def __init__(self):
    #     self.baseoutname = lambda : self.expname + self.baseName
    #     self.baseckptname = lambda: self.baseoutname() + 'ckpt'
    #     self.basedataname = lambda: self.baseoutname() + 'traindata'
    baseoutname = expname + baseName
    baseckptname = baseoutname + 'ckpt'
    basedataname = baseoutname + 'traindata'
    fineoutname = expname + fineName
    mrfoutname = expname + mrfName
    evaloutname = expname + evalName
    genoutname = expname + genName
    baseregoutname = expname + regName
    baseregckptname = baseregoutname + 'ckpt'
    fineckptname = fineoutname + 'ckpt'
    mrfckptname = mrfoutname + 'ckpt'
    evalckptname = evaloutname + 'ckpt'
    genckptname = genoutname + 'ckpt'
    baseregdataname = baseregoutname + 'traindata'
    finedataname = fineoutname + 'traindata'
    mrfdataname = mrfoutname + 'traindata'
    evaldataname = evaloutname + 'traindata'
    gendataname = genoutname + 'traindata'

    mrf_psz = 54


aliceConfig = myconfig()
aliceConfig.cachedir = os.path.join(localSetup.bdir, 'cacheAlice')
aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_20170925_cv.lbl')
def aliceexpname(dirname):
    return os.path.basename(os.path.dirname(dirname))

def alicegetexplist(L):
    return L['movieFilesAll'][0,:]

aliceConfig.getexpname = aliceexpname
aliceConfig.getexplist = alicegetexplist
aliceConfig.has_trx_file = True
aliceConfig.view = 0
aliceConfig.imsz = (180, 180)
aliceConfig.selpts = np.arange(0, 17)
aliceConfig.imgDim = 1

