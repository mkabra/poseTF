
# coding: utf-8

# In[ ]:




# In[ ]:

import os

name = 'janLeg'
split = True
scale = 2
rescale = 1
numscale = 3
pool_scale = 4
cropsz = 0

# sel_sz determines the patch size used for the final decision
# i.e., patch seen by the fc6 layer
# ideally as large as possible but limited by
# a) gpu memory size
# b) overfitting due to large number of variables.
sel_sz = 512/2/2
psz = sel_sz/(scale**(numscale-1))/rescale/pool_scale

imsz = (256,256)
cachedir = '/home/mayank/work/tensorflow/cache' + name + '/'
labelfile = '/home/mayank/work/tensorflow/janLegTracking/janLegData_20160301.mat'
# this label file has more data and includes the correction for vertical flipping
viddir = '/home/mayank/Dropbox/PoseEstimation/JanLegTracking'
ptn = '.*'
trainfilename = 'train_lmdb'
valfilename = 'val_lmdb'

valdatafilename = 'valdata'
valratio = 0.3

dist2pos = 5
label_blur_rad = 1.5
fine_label_blur_rad = 1.5

view = 0

base_learning_rate = 0.00001
fine_learning_rate = 0.0001
base_training_iters = 8000 # for a batch size of 32
fine_training_iters = 8000
joint_training_iters = 20000
gamma = 0.1
step_size = 200000
base_batch_size = 32
fine_batch_size = 32
display_step = 30

# Network Parameters
n_classes = 4 # 
dropout = 0.5 # Dropout, probability to keep units
nfilt = 128
nfcfilt = 256

fine_flt_sz = 5
fine_nfilt = 48
fine_sz = 36


map_size = 100000*imsz[0]*imsz[1]*3

outname = name
fineoutname = 'janLegFine'
jointoutname = 'janLegJoint'
ckptbasename = name + 'Baseckpt'
ckptfinename = name + 'Fineckpt'
ckptjointname = name + 'Jointckpt'
databasename = name + 'basetraindata'
datafinename = name + 'finetraindata'
datajointname = name + 'jointtraindata'

save_step = 1000
numTest = 100
maxckpt = 20

def getexpname(dirname):
    expname = os.path.basename(dirname)
    return expname

def getexplist(L):
    fname = 'vid{:d}files'.format(view+1)
    return L[fname]


