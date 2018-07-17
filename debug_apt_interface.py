import socket
import APT_interface as apt
import os
import shutil
import h5py
import logging
reload(apt)

lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'

split_file = '/home/mayank/work/poseTF/cache/apt_interface/multitarget_bubble_view0/test_leap/splitdata.json'

log = logging.getLogger()  # root logger
log.setLevel(logging.ERROR)

import deepcut.train
conf = apt.create_conf(lbl_file,0,'test_openpose_delete')
conf.splitType = 'predefined'
apt.create_tfrecord(conf, True, split_file=split_file)
from poseConfig import config as args
args.skip_db = True
apt.train_openpose(conf,args)

##
import deepcut.train
import  tensorflow as tf
tf.reset_default_graph
conf.batch_size = 1
pred_fn, model_file = deepcut.train.get_pred_fn(conf)
rfn, n= deepcut.train.get_read_fn(conf,'/home/mayank/work/poseTF/cache/apt_interface/multitarget_bubble_view0/test_deepcut/val_data.p')
A = apt.classify_db(conf, rfn, pred_fn, n)

##
import socket
import APT_interface as apt
import os
import shutil
import h5py
import logging

lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'

conf = apt.create_conf(lbl_file,view=0,name='test_openpose')
graph =  [ [1,2],[1,3],[2,5],[3,4],[1,6],[6,7],[6,8],[6,10],[8,9],[10,11],[5,12],[9,13],[6,14],[6,15],[11,16],[4,17]]
graph = [[g1-1, g2-1] for g1, g2 in graph]
conf.op_affinity_graph = graph

from poseConfig import config as args
# APT_interface.create_leap_db(conf,True)

# conf.batch_size = 32
# conf.rrange = 15
# conf.dl_steps = 2500

log = logging.getLogger()  # root logger
log.setLevel(logging.INFO)

args.skip_db = True
apt.train_openpose(conf, args)


##

import  h5py
import APT_interface as apt
lbl_file = '/home/mayank/work/poseTF/data/stephen/sh_trn4523_gt080618_made20180627_stripped.lbl'
from stephenHeadConfig import sideconf as conf
conf.labelfile = lbl_file

conf.cachedir = '/home/mayank/work/poseTF/cache/stephen'
from poseConfig import config as args
args.skip_db = False
apt.train_unet(conf,args)
##
import socket
import APT_interface
import os
import shutil
import h5py

lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'

conf = APT_interface.create_conf(lbl_file,view=0,name='test_leap')
# APT_interface.create_leap_db(conf,True)

conf.batch_size = 32
conf.rrange = 15
# conf.dl_steps = 2500

db_path = [os.path.join(conf.cachedir, 'leap_train.h5')]
db_path.append(os.path.join(conf.cachedir, 'leap_val.h5'))
import leap.training
reload(leap.training)
from leap.training import train_apt
train_apt(db_path,conf,'test_leap')

## test leap db creation
import APT_interface
lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'

conf = APT_interface.create_conf(lbl_file,view=0,name='stacked_hourglass')
APT_interface.create_leap_db(conf,True)

##


import socket
import APT_interface
import os
import shutil
import h5py

if socket.gethostname() == 'mayankWS':
    in_lbl_file = '/home/mayank/work/poseTF/data/alice/multitarget_bubble_expandedbehavior_20180425_local.lbl'
    lbl_file = '/home/mayank/work/poseTF/data/apt/alice_test_apt.lbl'
else:
    in_lbl_file = ''
    lbl_file = None


shutil.copyfile(in_lbl_file,lbl_file)
H = h5py.File(lbl_file,'r+')

H[H['trackerData'][1,0]]['sPrm'].keys()


