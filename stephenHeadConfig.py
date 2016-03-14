import os

# ----- Network parameters

scale = 2
rescale = 2
numscale = 3
pool_scale = 4
# sel_sz determines the patch size used for the final decision
# i.e., patch seen by the fc6 layer
# ideally as large as possible but limited by
# a) gpu memory size
# b) overfitting due to large number of variables.
sel_sz = 512/2/2
psz = sel_sz/(scale**(numscale-1))/rescale/pool_scale
dist2pos = 5
label_blur_rad = 1.5
fine_label_blur_rad = 1.5
n_classes = 5 # 
dropout = 0.5 # Dropout, probability to keep units
nfilt = 128
nfcfilt = 512
doBatchNorm = False
# ----- Fine Network parameters

fine_flt_sz = 5
fine_nfilt = 48
fine_sz = 36

# ----- MRF Network Parameters

#maxDPts = 104*2
#mrf_psz = (maxDPts/rescale)/pool_scale
#Above should not be determined automatically
baseIter4MRFTrain = 4000

# ----- Learning parameters

base_learning_rate = 0.0001
mrf_learning_rate = 0.0001
fine_learning_rate = 0.0001

base_training_iters = 8000
# when run iwth batch size of 32, best validation loss is achieved at 8000 iters 
# for FlyHeadStephenCuratedData.mat -- Feb 11, 2016 Mayank
fine_training_iters = 20000
mrf_training_iters = 20000
gamma = 0.1
step_size = 200000
batch_size = 32
display_step = 30
numTest = 100
# fine_batch_size = 8


# ----- Data parameters

split = True
view = 0
cropsz = 200
imsz = (624,624) # This is after cropping. Orig is 1024x1024
map_size = 100000*psz**2*3

cachedir = '/home/mayank/work/tensorflow/cacheHead/'
labelfile = '/home/mayank/work/tensorflow/headTracking/FlyHeadStephenCuratedData.mat'
viddir = '/home/mayank/Dropbox/PoseEstimation/Stephen'
ptn = 'fly_000[0-9]'
trainfilename = 'train_lmdb'
valfilename = 'val_lmdb'
valdatafilename = 'valdata'
valratio = 0.3


# ----- Save parameters

save_step = 1000
maxckpt = 20
outname = 'headBase'
fineoutname = 'headFine'
mrfoutname = 'headMRF'
ckptbasename = 'headBaseckpt'
ckptfinename = 'headFineckpt'
ckptmrfname = 'headMRFckpt'
databasename = 'headbasetraindata'
datafinename = 'headfinetraindata'
datamrfname = 'headMRFtraindata'

# ----- project specific functions

def getexpname(dirname):
    dirname = os.path.normpath(dirname)
    dir_parts = dirname.split(os.sep)
    expname = dir_parts[-6] + "!" + dir_parts[-3]
    return expname

def getexplist(L):
    fname = 'vid{:d}files'.format(view+1)
    return L[fname]
