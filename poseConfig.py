from __future__ import division

# coding: utf-8

# In[ ]:

from builtins import object
from past.utils import old_div
import os
import re
import localSetup
import numpy as np


class config(object):
    # ----- Names

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
    dist2pos = 5
    label_blur_rad = 3  # 1.5
    fine_label_blur_rad = 1.5
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
    adjustContrast = True
    clahegridsize = 20
    normalize_img_mean = True
    normalize_batch_mean = False
    perturb_color = False

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
    max_n_animals = 1

    # ----- UNet params
    unet_rescale = 1

    # ----- MDN params
    mdn_min_sigma = 3. # this should just be maybe twice the cell size??
    mdn_max_sigma = 4.
    mdn_logit_eps_training = 0.001
    mdn_extra_layers = 0

    # ----- Save parameters

    save_step = 2000
    maxckpt = 30
    def set_exp_name(self, exp_name):
        self.expname = exp_name
        self.baseoutname = self.expname + self.baseName
        self.baseckptname = self.baseoutname + 'ckpt'
        self.basedataname = self.baseoutname + 'traindata'
        self.fineoutname = self.expname + self.fineName
        self.fineckptname = self.fineoutname + 'ckpt'
        self.finedataname = self.fineoutname + 'traindata'
        self.mrfoutname = self.expname + self.mrfName
        self.mrfckptname = self.mrfoutname + 'ckpt'
        self.mrfdataname = self.mrfoutname + 'traindata'
        # self.evaloutname = self.expname + evalName
        # self.evalckptname = self.evaloutname + 'ckpt'
        # self.evaldataname = self.evaloutname + 'traindata'
        # self.genoutname = self.expname + genName
        # self.genckptname = self.genoutname + 'ckpt'
        # self.gendataname = self.genoutname + 'traindata'
        self.baseregoutname = self.expname + self.regName
        self.baseregckptname = self.baseregoutname + 'ckpt'
        self.baseregdataname = self.baseregoutname + 'traindata'

    # baseoutname = expname + baseName
    # baseckptname = baseoutname + 'ckpt'
    # basedataname = baseoutname + 'traindata'
    # fineoutname = expname + fineName
    # mrfoutname = expname + mrfName
    # evaloutname = expname + evalName
    # genoutname = expname + genName
    # baseregoutname = expname + regName
    # baseregckptname = baseregoutname + 'ckpt'
    # fineckptname = fineoutname + 'ckpt'
    # mrfckptname = mrfoutname + 'ckpt'
    # evalckptname = evaloutname + 'ckpt'
    # genckptname = genoutname + 'ckpt'
    # baseregdataname = baseregoutname + 'traindata'
    # finedataname = fineoutname + 'traindata'
    # mrfdataname = mrfoutname + 'traindata'
    # evaldataname = evaloutname + 'traindata'
    # gendataname = genoutname + 'traindata'


# -- alice fly --

aliceConfig = config()
aliceConfig.cachedir = os.path.join(localSetup.bdir, 'cache','alice')
#aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_20170925_cv.lbl')
aliceConfig.labelfile = os.path.join(localSetup.bdir,'data','alice','multitarget_bubble_20180107.lbl')
def alice_exp_name(dirname):
    return os.path.basename(os.path.dirname(dirname))

def alice_get_exp_list(L):
    return L['movieFilesAll'][0,:]

aliceConfig.getexpname = alice_exp_name
aliceConfig.getexplist = alice_get_exp_list
aliceConfig.has_trx_file = True
aliceConfig.view = 0
aliceConfig.imsz = (180, 180)
aliceConfig.selpts = np.arange(0, 17)
aliceConfig.imgDim = 1
aliceConfig.n_classes = len(aliceConfig.selpts)
aliceConfig.splitType = 'frame'
aliceConfig.set_exp_name('aliceFly')
aliceConfig.trange = 5
aliceConfig.nfcfilt = 128
aliceConfig.sel_sz = 144
aliceConfig.num_pools = 1
aliceConfig.dilation_rate = 2
aliceConfig.pool_scale = aliceConfig.pool_stride**aliceConfig.num_pools
aliceConfig.psz = aliceConfig.sel_sz / 4 / aliceConfig.pool_scale / aliceConfig.dilation_rate
aliceConfig.valratio = 0.25
aliceConfig.mdn_min_sigma = 70.
aliceConfig.mdn_max_sigma = 70.
aliceConfig.adjustContrast = False
aliceConfig.clahegridsize = 10
aliceConfig.brange = [0,0]
aliceConfig.crange = [1.,1.]
aliceConfig.mdn_extra_layers = 1
aliceConfig.normalize_img_mean = False
# aliceConfig.mdn_groups = [range(11),[11],[12],[13],[14],[15],[16]]
#aliceConfig.mdn_groups = [range(13)+[15,16],[13],[14]]
aliceConfig.mdn_groups = [range(17)]
# -- felipe bees --

felipeConfig = config()
felipeConfig.cachedir = os.path.join(localSetup.bdir, 'cache','felipe')
felipeConfig.labelfile = os.path.join(localSetup.bdir,'data','felipe','doesnt_exist.lbl')
def felipe_exp_name(dirname):
    return dirname

def felipe_get_exp_list(L):
    return 0

felipeConfig.getexpname = felipe_exp_name
felipeConfig.getexplist = felipe_get_exp_list
felipeConfig.view = 0
felipeConfig.imsz = (300, 300)
felipeConfig.selpts = np.arange(0, 5)
felipeConfig.imgDim = 3
felipeConfig.n_classes = len(felipeConfig.selpts)
felipeConfig.splitType = 'frame'
felipeConfig.set_exp_name('felipeBees')
felipeConfig.trange = 20
felipeConfig.nfcfilt = 128
felipeConfig.sel_sz = 144
felipeConfig.num_pools = 2
felipeConfig.dilation_rate = 1
felipeConfig.pool_scale = felipeConfig.pool_stride**felipeConfig.num_pools
felipeConfig.psz = felipeConfig.sel_sz / 4 / felipeConfig.pool_scale / felipeConfig.dilation_rate


##  -- felipe multi bees

# -- felipe bees --

felipe_config_multi = config()
felipe_config_multi.cachedir = os.path.join(localSetup.bdir, 'cache', 'felipe_m')
felipe_config_multi.labelfile = os.path.join(localSetup.bdir, 'data', 'felipe_m', 'doesnt_exist.lbl')
def felipe_exp_name(dirname):
    return dirname

def felipe_get_exp_list(L):
    return 0

felipe_config_multi.getexpname = felipe_exp_name
felipe_config_multi.getexplist = felipe_get_exp_list
felipe_config_multi.view = 0
felipe_config_multi.imsz = (360, 380)
felipe_config_multi.selpts = np.array([1, 3, 4])
felipe_config_multi.imgDim = 3
felipe_config_multi.n_classes = len(felipe_config_multi.selpts)
felipe_config_multi.splitType = 'frame'
felipe_config_multi.set_exp_name('felipeBeesMulti')
felipe_config_multi.trange = 20
felipe_config_multi.nfcfilt = 128
felipe_config_multi.sel_sz = 256
felipe_config_multi.num_pools = 2
felipe_config_multi.dilation_rate = 1
felipe_config_multi.pool_scale = felipe_config_multi.pool_stride ** felipe_config_multi.num_pools
felipe_config_multi.psz = felipe_config_multi.sel_sz / 4 / felipe_config_multi.pool_scale / felipe_config_multi.dilation_rate
felipe_config_multi.max_n_animals = 17
