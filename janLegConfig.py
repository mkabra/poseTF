
# coding: utf-8

# In[ ]:




# In[ ]:

import os

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
n_classes = 4 # 
nfilt = 128
nfcfilt = 256

# ----- Fine Network parameters
fine_label_blur_rad = 1.5
fine_flt_sz = 5
fine_nfilt = 48
fine_sz = 36

# ----- MRF Network Parameters
baseIter4MRFTrain = 4000


# ----- Learning parameters

dropout = 0.5 # Dropout, probability to keep units
base_learning_rate = 0.00001
fine_learning_rate = 0.0001
base_training_iters = 8000 # for a batch size of 32
fine_training_iters = 8000
joint_training_iters = 20000
gamma = 0.1
step_size = 200000
batch_size = 32
fine_batch_size = 32
display_step = 30
numTest = 100


# ----- Data parameters

valratio = 0.3
imsz = (256,256)
split = True
view = 0
cropsz = 0
map_size = 100000*imsz[0]*imsz[1]*3
cachedir = '/home/mayank/work/tensorflow/cachejanLeg/'
labelfile = '/home/mayank/work/tensorflow/janLegTracking/janLegData_20160301.mat'
# this label file has more data and includes the correction for vertical flipping
viddir = '/home/mayank/Dropbox/PoseEstimation/JanLegTracking'
ptn = '.*'
trainfilename = 'train_lmdb'
valfilename = 'val_lmdb'
valdatafilename = 'valdata'


# ----- Save parameters

save_step = 1000
maxckpt = 20
name = 'janLeg'
outname = name+'Base'
fineoutname = 'janLegFine'
mrfoutname = 'janLegMRF'
ckptbasename = name + 'Baseckpt'
ckptfinename = name + 'Fineckpt'
ckptmrfname = name + 'MRFckpt'
databasename = name + 'basetraindata'
datafinename = name + 'finetraindata'
datamrfname = name + 'mrftraindata'
datajointname = name + 'jointtraindata'


def getexpname(dirname):
    expname = os.path.basename(dirname)
    return expname

def getexplist(L):
    fname = 'vid{:d}files'.format(view+1)
    return L[fname]


