from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[13]:

from builtins import range
from past.utils import old_div
import os
import re
import glob
import sys
import argparse
from subprocess import call
import stat
import h5py
# import hdf5storage
import scipy.io as sio
import hdf5storage
import localSetup
import multiResData
import PoseUNet
import numpy as np
import tensorflow as tf
import cv2
from cvc import cvc
import datetime
import movies

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    From: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def read_entry(x):
    if type(x) is h5py._hl.dataset.Dataset:
        return x[0,0]
    else:
        return x

def read_string(x):
    if type(x) is h5py._hl.dataset.Dataset:
        return ''.join(chr(c) for c in x)
    else:
        return x

def has_trx_file(x):
    if type(x) is h5py._hl.dataset.Dataset:
        if len(x.shape) == 1:
            has_trx = False
        else:
            has_trx = True
    else:
        if x.size == 0:
            has_trx = False
        else:
            has_trx = True
    return  has_trx

def create_conf(lblfile, view):
    try:
        H = loadmat(lblfile)
    except NotImplementedError:
        print('Label file is in v7.3 format. Loading using h5py')
        H = h5py.File(lblfile)

    from poseConfig import config
    conf = config()
    conf.n_classes = int(read_entry(H['cfg']['NumLabelPoints']))
    proj_name =  read_string(H['projname']) + '_view{}'.format(view)
    conf.view = view
    conf.set_exp_name(proj_name)
    # conf.cacheDir = read_string(H['cachedir'])
    dt_params = H['trackerDeepData']['sPrm']
    conf.cachedir = os.path.join(read_string(dt_params['CacheDir']),proj_name)
    # conf.cachedir = os.path.join(localSetup.bdir, 'cache', proj_name)
    conf.has_trx_file = has_trx_file(H[H['trxFilesAll'][0,0]])
    conf.imgDim = int(read_entry(dt_params['NChannels']))
    conf.selpts = np.arange(conf.n_classes)
    vid_nr = int(read_entry(H[H['movieInfoAll'][0,0]]['info']['nr']))
    vid_nc = int(read_entry(H[H['movieInfoAll'][0,0]]['info']['nc']))
    im_nr = int(read_entry(dt_params['Size']))
    conf.imsz = (im_nr,im_nr)
    crop_locX = int(read_entry(dt_params['CropX_view{}'.format(view+1)]))
    crop_locY = int(read_entry(dt_params['CropY_view{}'.format(view+1)]))
    conf.cropLoc = {(vid_nr,vid_nc): [crop_locX, crop_locY]}
    conf.labelfile = lblfile
    conf.sel_sz = min(conf.imsz)
    conf.unet_rescale = int(read_entry(dt_params['scale']))
    conf.adjustContrast = int(read_entry(dt_params['adjustContrast']))>0.5
    conf.normalize_img_mean= int(read_entry(dt_params['normalize']))>0.5

    return  conf

def datetime2matlabdn(dt=datetime.datetime.now()):
   mdn = dt + datetime.timedelta(days = 366)
   frac_seconds = (dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
   frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
   return mdn.toordinal() + frac_seconds + frac_microseconds

def train(lblfile, nviews):

    for view in range(nviews):
        conf = create_conf(lblfile, view)
        conf.view = view
        if conf.has_trx_file:
            multiResData.createTFRecordFromLblWithTrx(conf,False)
        else:
            multiResData.createTFRecordFromLbl(conf,False)
        tf.reset_default_graph()
        self = PoseUNet.PoseUNet(conf)
        self.train_unet(False,1)

def classify(lbl_file, n_views, mov_file, trx_file, start_frame, end_frame, out_file, skip_rate):
    # print(mov_file)

    for view in range(n_views):
        conf = create_conf(lbl_file, view)
        tf.reset_default_graph()
        self = PoseUNet.PoseUNet(conf)
        sess = self.init_net_meta(0,True)

        cap = movies.Movie(mov_file[view],interactive=False)
        n_frames = int(cap.get_n_frames())
        height = int(cap.get_height())
        width = int(cap.get_width())
        cur_end_frame = end_frame
        if end_frame < 0:
            cur_end_frame = n_frames
        cap.close()

        if trx_file is not None:
            predList = self.classify_movie_trx(mov_file[view],trx_file[view], sess,end_frame, start_frame)
            temp_predScores = predList[1]
            predScores = -1 * np.ones((n_frames,) + temp_predScores.shape[1:])
            # predScores[start_frame:cur_end_frame, ...] = temp_predScores
            # dummy predscores for now.

            temp_pred_locs = predList
            pred_locs = np.nan * np.ones((n_frames,) + temp_pred_locs.shape[1:])
            pred_locs[start_frame:cur_end_frame, ...] = temp_pred_locs
            pred_locs = pred_locs.transpose([2,3,0,1])
            tgt = np.arange(pred_locs.shape[-1])+1
        else:
            predList = self.classify_movie(mov_file[view],sess,end_frame, start_frame)

            rescale = conf.unet_rescale
            orig_crop_loc = conf.cropLoc[(height,width)]
            crop_loc = [int(x/rescale) for x in orig_crop_loc]
            end_pad = [int((height-conf.imsz[0])/rescale)-crop_loc[0],int((width-conf.imsz[1])/rescale)-crop_loc[1]]

            pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0)]
            temp_predScores = np.pad(predList[1],pp,mode='constant',constant_values=-1.)
            predScores = np.zeros((n_frames,)+temp_predScores.shape[1:])
            predScores[start_frame:cur_end_frame,...] = temp_predScores

            temp_pred_locs = predList[0]
            temp_pred_locs[:,:,0] += orig_crop_loc[1]
            temp_pred_locs[:,:,1] += orig_crop_loc[0]
            pred_locs = np.nan*np.ones((n_frames,)+temp_pred_locs.shape[1:])
            pred_locs[start_frame:cur_end_frame,...] = temp_pred_locs
            pred_locs = pred_locs.transpose([1,2,0])
            tgt = np.arange(1)+1

        ts_shape = pred_locs.shape[0:1] + pred_locs.shape[2:]
        ts = np.ones(ts_shape)*datetime2matlabdn()
        tag = np.zeros(ts.shape).astype('bool')
        hdf5storage.savemat(out_file + '_view_{}'.format(view)+ '.trk', {'pTrk': pred_locs, 'pTrkTS':ts,'expname':mov_file[view],'pTrkiTgt':tgt,'pTrkTag':tag}, appendmat=False, truncate_existing=True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("lbl_file",
                        help="path to lbl file")
    subparsers = parser.add_subparsers(help='train or track', dest='sub_name')

    parser_train = subparsers.add_parser('train',help='Train the detector')
    # parser_train.add_argument('-cache',dest='cache_dir',
    #                           help='cache dir for training')

    parser_classify = subparsers.add_parser('track',help='Track a movie')
    parser_classify.add_argument("-mov", dest="mov",
                        help="movie to track",required=True,nargs='+')
    parser_classify.add_argument("-trx", dest="trx",
                        help='trx file for above movie',default=None,nargs='*')
    parser_classify.add_argument('-start_frame',dest='start_frame', help='start tracking from this frame',type=int,default=0)
    parser_classify.add_argument('-end_frame',dest='end_frame', help='end frame for tracking',type=int,default=-1)
    parser_classify.add_argument('-skip_rate', dest='skip',help='frames to skip while tracking',default=1,type=int)
    parser_classify.add_argument('-out_file', dest='out_file',help='file to save tracking results to',required=True)

    print(argv)
    args = parser.parse_args(argv)

    lblfile = args.lbl_file
    try:
        H = loadmat(lblfile)
    except NotImplementedError:
        print('Label file is in v7.3 format. Loading using h5py')
        H = h5py.File(lblfile)
    nviews = int(read_entry(H['cfg']['NumViews']))

    if args.sub_name == 'train':
        train(lblfile, nviews)
    elif args.sub_name == 'track':
        classify(lblfile, nviews, args.mov, args.trx, args.start_frame, args.end_frame, args.out_file,args.skip )


if __name__ == "__main__":
    main(sys.argv[1:])

