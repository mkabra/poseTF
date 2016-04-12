
# coding: utf-8

# In[ ]:




# In[ ]:

import os,socket
import localSetup

class myconfig(object):
    expname = 'janLeg'
    baseName = 'Base'
    fineName = 'Fine'
    mrfName = 'MRF'
    acName = 'AC'
    
    # ----- Network Parameters

    scale = 2
    rescale = 1
    numscale = 3
    pool_scale = 4
    # sel_sz determines the patch size used for the final decision
    # i.e., patch seen by the fc6 layer
    # ideally as large as possible but limited by
    # a) gpu memory size
    # b) overfitting due to large number of variables.
    sel_sz = 512/2/2
    psz = sel_sz/(scale**(numscale-1))/rescale/pool_scale
    label_blur_rad = 1.5
    fine_label_blur_rad = 1.5
    n_classes = 4 
    dropout = 0.5
    nfilt = 128
    nfcfilt = 512
    doBatchNorm = True
    useMRF = False
    useAC = True
    useHoldout = False

    # ----- Fine Network parameters
    fine_flt_sz = 5
    fine_nfilt = 48
    fine_sz = 48

    # ----- MRF Network Parameters
    baseIter4MRFTrain = 5000
    baseIter4ACTrain = 5000


    # ----- Learning parameters

    base_learning_rate = 0.0003
    mrf_learning_rate = 0.00001
    ac_learning_rate = 0.0003
    fine_learning_rate = 0.0003
    base_training_iters = 5000 # for a batch size of 32
    fine_training_iters = 3000
    mrf_training_iters = 3000
    ac_training_iters = 5000
    gamma = 0.1
    step_size = 200000
    batch_size = 16
    display_step = 30
    numTest = 100


    # ----- Data parameters

    imsz = (256,256)
    split = True
    view = 0
    l1_cropsz = 0
    map_size = 100000*imsz[0]*imsz[1]*3
    cropLoc = {(256,256):[0,0]}
    cachedir = os.path.join(localSetup.bdir,'cachejanLeg/')
    labelfile = os.path.join(localSetup.bdir,'janLegTracking','janLegData_20160301.mat')
    # this label file has more data and includes the correction for vertical flipping
    ptn = '.*'
    trainfilename = 'train_lmdb'
    valfilename = 'val_lmdb'
    valdatafilename = 'valdata'
    valratio = 0.3
    holdouttrain = 'holdouttrain_lmdb'
    holdouttest = 'holdouttest_lmdb'
    holdoutratio = 0.8


    # ----- Save parameters

    save_step = 500
    maxckpt = 20
    baseoutname = expname + baseName
    fineoutname = expname + fineName
    mrfoutname = expname + mrfName
    acoutname = expname + acName
    baseckptname = baseoutname + 'ckpt'
    fineckptname = fineoutname + 'ckpt'
    mrfckptname = mrfoutname + 'ckpt'
    acckptname = acoutname + 'ckpt'
    basedataname = baseoutname + 'traindata'
    finedataname = fineoutname + 'traindata'
    mrfdataname = mrfoutname + 'traindata'
    acdataname = acoutname + 'traindata'


    def getexpname(self,dirname):
        expname = os.path.basename(dirname)
        return expname

    def getexplist(self,L):
        fname = 'vid{:d}files'.format(view+1)
        return L[fname]

conf = myconfig()

