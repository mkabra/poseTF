split = True
scale = 2
rescale = 2
numscale = 3
pool_scale = 4
cropsz = 200

# sel_sz determines the patch size used for the final decision
# i.e., patch seen by the fc6 layer
# ideally as large as possible but limited by
# a) gpu memory size
# b) overfitting due to large number of variables.
sel_sz = 512/2
psz = sel_sz/(scale**(numscale-1))/rescale

cachedir = '/home/mayank/work/tensorflow/cacheHead/'
labelfile = '/home/mayank/work/tensorflow/headTracking/FlyHeadStephenTestData_20150813.mat'
viddir = '/home/mayank/Dropbox/PoseEstimation/Stephen'
ptn = 'fly_000[0-9]'
trainfilename = 'train_lmdb'
valfilename = 'val_lmdb'



valdatafilename = 'valdata'
valratio = 0.3

dist2pos = 5
label_blur_rad = 8.

view = 1

learning_rate = 0.0001
training_iters = 20000
batch_size = 16
display_step = 30

# Network Parameters
n_classes = 5 # 
dropout = 0.5 # Dropout, probability to keep units
nfilt = 128
nfcfilt = 512

map_size = 100000*psz**2*3

outname = 'head'
save_step = 4000
numTest = 400
imsz = (624,624)

def getexpname(dirname):
    dirname = os.path.normpath(dirname)
    dir_parts = dirname.split(os.sep)
    expname = dir_parts[-6] + "!" + dir_parts[-3]
    return expname

def getexplist(L):
    fname = 'vid{:d}files'.format(view+1)
    return L[fname]