from __future__ import division
from __future__ import print_function

from builtins import range
import os
import sys
import argparse
import h5py
import scipy.io as sio
import hdf5storage
import multiResData
import PoseUNet
import numpy as np
import tensorflow as tf
import datetime
import movies
import math
import PoseTools
from multiResData import *
import json
from random import sample

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    From: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True,appendmat=False)
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
        return x[0, 0]
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
    return has_trx


def create_conf(lbl_file, view, name):
    try:
        H = loadmat(lbl_file)
    except NotImplementedError:
        print('Label file is in v7.3 format. Loading using h5py')
        H = h5py.File(lbl_file,'r')

    from poseConfig import config
    conf = config()
    conf.n_classes = int(read_entry(H['cfg']['NumLabelPoints']))
    proj_name = read_string(H['projname']) + '_view{}'.format(view)
    conf.view = view
    conf.set_exp_name(proj_name)
    # conf.cacheDir = read_string(H['cachedir'])
    dt_params = H['trackerDeepData']['sPrm']
    conf.cachedir = os.path.join(read_string(dt_params['CacheDir']), proj_name, name)
    if not os.path.exists(os.path.split(conf.cachedir)[0]):
        os.mkdir(os.path.split(conf.cachedir)[0])
    # conf.cachedir = os.path.join(localSetup.bdir, 'cache', proj_name)
    conf.has_trx_file = has_trx_file(H[H['trxFilesAll'][0, 0]])
    conf.imgDim = int(read_entry(dt_params['NChannels']))
    conf.selpts = np.arange(conf.n_classes)
    vid_nr = int(read_entry(H[H['movieInfoAll'][0, 0]]['info']['nr']))
    vid_nc = int(read_entry(H[H['movieInfoAll'][0, 0]]['info']['nc']))
    width = int(read_entry(dt_params['sizex']))
    height = int(read_entry(dt_params['sizey']))
    conf.imsz = (height, width)
    crop_locX = int(read_entry(dt_params['CropX_view{}'.format(view + 1)]))
    crop_locY = int(read_entry(dt_params['CropY_view{}'.format(view + 1)]))
    conf.cropLoc = {(vid_nr, vid_nc): [crop_locY, crop_locX]}
    conf.labelfile = lbl_file
    conf.sel_sz = min(conf.imsz)
    conf.unet_rescale = int(read_entry(dt_params['scale']))
    conf.adjustContrast = int(read_entry(dt_params['adjustContrast'])) > 0.5
    conf.normalize_img_mean = int(read_entry(dt_params['normalize'])) > 0.5
    try:
        conf.flipud = int(read_entry(dt_params['flipud'])) > 0.5
    except KeyError:
        pass
    try:
        conf.unet_steps= int(read_entry(dt_params['unet_steps']))
    except KeyError:
        pass
    try:
        conf.save_td_step = read_entry(dt_params['save_td_step'])
    except KeyError:
        pass
    try:
        bb = read_entry(dt_params['brange'])
        conf.brange = [-bb,bb]
    except KeyError:
        pass
    try:
        bb = read_entry(dt_params['crange'])
        conf.crange = [1-bb,1+bb]
    except KeyError:
        pass
    try:
        bb = read_entry(dt_params['trange'])
        conf.trange = bb
    except KeyError:
        pass
    try:
        bb = read_entry(dt_params['rrange'])
        conf.rrange = bb
    except KeyError:
        pass

    return conf


def datetime2matlabdn(dt=datetime.datetime.now()):
    mdn = dt + datetime.timedelta(days=366)
    frac_seconds = (dt - datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds

def db_from_lbl(conf, out_fns, split=True, split_file=None):
    # outputs is a list of functions. The first element writes
    # to the training dataset while the second one write to the validation
    # dataset. If split is False, second element is not used and all data is
    # outputted to training dataset
    # the function will be give a list with:
    # 0: img,
    # 1: locations as a numpy array.
    # 2: information list [expid, frame number, trxid]
    # the function returns a list of [expid, frame_number and trxid] showing
    #  how the data was split between the two datasets.

    local_dirs, _ = multiResData.find_local_dirs(conf)
    lbl = h5py.File(conf.labelfile)
    view = conf.view
    flipud = conf.flipud
    npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
    sel_pts = int(view * npts_per_view) + conf.selpts

    splits = [[],[]]
    count = 0
    val_count = 0

    mov_split = None
    predefined = None
    if conf.splitType is 'predefined':
        assert split_file is not None, 'File for defining splits is not given'
        predefined = json.load(split_file)
    elif conf.splitType is 'movie':
        nexps = len(local_dirs)
        mov_split = sample(list(range(nexps)), int(nexps * conf.valratio))
        predefined = None
    elif conf.splitType is 'trx':
        assert conf.has_trx_file, 'Train/Validation was selected to be trx but the project has no trx files'

    for ndx, dir_name in enumerate(local_dirs):

        exp_name = conf.getexpname(dir_name)
        cur_pts = trx_pts(lbl, ndx)
        try:
            cap = movies.Movie(local_dirs[ndx])
        except ValueError:
            raise IOError(local_dirs[ndx] + ' is missing')

        if conf.has_trx_file:
            trx_files = multiResData.get_trx_files(lbl, local_dirs)
            trx = sio.loadmat(trx_files[ndx])['trx'][0]
            n_trx = len(trx)
            trx_split = np.random.random(n_trx) < conf.valratio
        else:
            trx = [None]
            n_trx = 1
            trx_split = None
            cur_pts = cur_pts[np.newaxis,...]

        for trx_ndx in range(n_trx):

            frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx)
            cur_trx = trx[trx_ndx]
            for fnum in frames:
                if not check_fnum(fnum, cap, exp_name, ndx):
                    continue

                info = [ndx, fnum, trx_ndx]
                cur_out = multiResData.get_cur_env( out_fns, split, conf, info,
                    mov_split, trx_split=trx_split, predefined=predefined)

                frame_in, cur_loc = multiResData.get_patch(
                        cap, fnum, conf, cur_pts[trx_ndx, fnum, :, sel_pts],cur_trx=cur_trx,flipud=flipud)
                cur_out([frame_in, cur_loc, info])

                if cur_out is out_fns[1] and split:
                    val_count += 1
                    splits[1].append(info)
                else:
                    count += 1
                    splits[0].append(info)

        cap.close()  # close the movie handles
        print('Done %d of %d movies, train count:%d val count:%d' % (ndx + 1, len(local_dirs), count, val_count))

    print('%d,%d number of pos examples added to the db and valdb' % (count, val_count))
    lbl.close()
    return splits

def tf_serialize(data):
    frame_in, cur_loc, info = data
    rows, cols, depth = frame_in.shape
    expid, fnum, trxid = info
    image_raw = frame_in.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': int64_feature(rows),
        'width': int64_feature(cols),
        'depth': int64_feature(depth),
        'trx_ndx': int64_feature(trxid),
        'locs': float_feature(cur_loc.flatten()),
        'expndx': float_feature(expid),
        'ts': float_feature(fnum),
        'image_raw': bytes_feature(image_raw)}))

    return example.SerializeToString()

def create_tfrecord(conf, split=True, split_file=None):
    # function showing how to use db_from_lbl for tfrecords
    if not os.path.exists(conf.cachedir):
        os.mkdir(conf.cachedir)
    envs = multiResData.create_envs(conf, split)
    out_fns = []
    out_fns.append(lambda data: envs[0].write(tf_serialize(data)))
    out_fns.append(lambda data: envs[1].write(tf_serialize(data)))
    splits = db_from_lbl(conf,out_fns, split, split_file)
    envs[0].close()
    envs[1].close() if split else None
    with open(os.path.join(conf.cachedir,'splitdata.json'),'w') as f:
        json.dump(splits, f)


def convert_to_orig(base_locs,conf,cur_trx,trx_fnum_start,all_f, sz,nvalid):
    # converts locs in cropped image to original image.
    if conf.has_trx_file:
        hsz_p = conf.imsz[0] / 2  # half size for pred
        base_locs_orig = np.zeros(base_locs.shape)
        for ii in range(nvalid):
            trx_fnum = trx_fnum_start + ii
            x = int(round(cur_trx['x'][0, trx_fnum])) - 1
            y = int(round(cur_trx['y'][0, trx_fnum])) - 1
            # -1 for 1-indexing in matlab and 0-indexing in python
            theta = cur_trx['theta'][0, trx_fnum]
            assert conf.imsz[0] == conf.imsz[1]
            tt = -theta - math.pi / 2
            R = [[np.cos(tt), -np.sin(tt)], [np.sin(tt), np.cos(tt)]]
            curlocs = np.dot(base_locs[ii, :, :] - [hsz_p, hsz_p],R) + [x,y]
            base_locs_orig[ii, ...] = curlocs
    else:
        orig_crop_loc = conf.cropLoc[sz]
        base_locs_orig = base_locs.copy()
        base_locs_orig[:, :, 0] += orig_crop_loc[1]
        base_locs_orig[:, :, 1] += orig_crop_loc[0]

    return base_locs_orig


def create_cv_split_files(conf, n_splits=3):
    # creates json files for the xv splits
    local_dirs, _ = multiResData.find_local_dirs(conf)
    lbl = h5py.File(conf.labelfile)

    mov_info = []
    trx_info = []
    n_labeled_frames = 0
    for ndx, dir_name in enumerate(local_dirs):
        if conf.has_trx_file:
            trx_files = multiResData.get_trx_files(lbl, local_dirs)
            trx = sio.loadmat(trx_files[ndx])['trx'][0]
            n_trx = len(trx)
        else:
            n_trx = 1

        cur_mov_info = []
        for trx_ndx in range(n_trx):
            frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx)
            mm = [ndx] * frames.size
            tt = [trx_ndx] * frames.size
            cur_trx_info = list(zip(mm,tt,frames.tolist()))
            trx_info.append(cur_trx_info)
            cur_mov_info.extend(cur_trx_info)
            n_labeled_frames += frames.size
        mov_info.append(cur_mov_info)
    lbl.close()

    lbls_per_fold = n_labeled_frames/n_splits

    imbalance = True
    for retry in range(10):
        per_fold = np.zeros([n_splits])
        splits = [[] for i in range(n_splits)]

        if conf.splitType is 'movie':
            for ndx in range(len(local_dirs)):
                valid_folds = np.where(per_fold<lbls_per_fold)[0]
                cur_fold = np.random.choice(valid_folds)
                splits[cur_fold].extend(mov_info[ndx])
                per_fold[cur_fold] += len(mov_info[ndx])

        elif conf.splitType is 'trx':
            for tndx in range(len(trx_info)):
                valid_folds = np.where(per_fold<lbls_per_fold)[0]
                cur_fold = np.random.choice(valid_folds)
                splits[cur_fold].extend(trx_info[tndx])
                per_fold[cur_fold] += len(trx_info[tndx])

        elif conf.splitType is 'frames':
            for ndx in range(len(local_dirs)):
                for mndx in range(len(mov_info[ndx])):
                    valid_folds = np.where(per_fold<lbls_per_fold)[0]
                    cur_fold = np.random.choice(valid_folds)
                    splits[cur_fold].extend(mov_info[ndx][mndx:mndx+1])
                    per_fold[cur_fold] += 1

            else:
                raise ValueError('splitType has to be either movie trx or frames')

        imbalance = (per_fold.max()-per_fold.min()) > float(lbls_per_fold)/3
        if not imbalance:
            break

    if imbalance:
        print('Couldnt find a valid spilt for split type:{} even after 10 retries.'.format(conf.splitType))
        print('Try changing the split type')
        return None

    all_train = []
    for ndx in range(n_splits):
        cur_train = []
        for idx, cur_split in enumerate(splits):
            if idx is not ndx:
                cur_train.extend(cur_split)
        all_train.append(cur_train)
        with open(os.path.join(conf.cachedir,'cv_split_fold_{}.json'.format(ndx)),'w'):
            json.dump([cur_train,splits[ndx]])

    return all_train, splits


def classify_movie_old(conf, pred_fn, mov_file, out_file, trx_file=None, start_frame=0, end_frame=-1, skip_rate=1):

    def write_trk(pred_locs_in,n_done):
        pred_locs = pred_locs_in.copy()
        pred_locs = pred_locs.transpose([2, 3, 0, 1])
        tgt = np.arange(pred_locs.shape[-1]) + 1
        if not conf.has_trx_file:
            pred_locs = pred_locs[...,0]
        ts_shape = pred_locs.shape[0:1] + pred_locs.shape[2:]
        ts = np.ones(ts_shape) * datetime2matlabdn()
        tag = np.zeros(ts.shape).astype('bool')
        tracked_shape = pred_locs.shape[2:]
        tracked = np.zeros(tracked_shape)
        tracked[:n_done,:] = 1
        hdf5storage.savemat(out_file,
                            {'pTrk': pred_locs, 'pTrkTS': ts, 'expname': mov_file, 'pTrkiTgt': tgt,
                             'pTrkTag': tag,'pTrkFrm':tracked}, appendmat=False, truncate_existing=True)

    cap = movies.Movie(mov_file)
    sz = (cap.get_height(), cap.get_width())
    n_frames = int(cap.get_n_frames())
    if conf.has_trx_file:
        try:
            T = sio.loadmat(trx_file)['trx'][0]
        except ValueError:
            raise IOError('Trx file {} is not a mat file'.format(trx_file))
        n_trx = len(T)
        end_frames = np.array([x['endframe'][0, 0] for x in T])
        first_frames = np.array([x['firstframe'][0, 0] for x in T]) - 1  # for converting from 1 indexing to 0 indexing
    else:
        T = [None,]
        n_trx = 1
        end_frames = np.array([n_frames])
        first_frames = np.array([0])

    if end_frame < 0:
        end_frame = end_frames.max()
    if end_frame > end_frames.max():
        end_frame = end_frames.max()
    if start_frame > end_frame:
        return None

    max_n_frames = end_frame - start_frame
    pred_locs = np.zeros([max_n_frames, n_trx, conf.n_classes, 2])
    pred_locs[:] = np.nan

    bsize = conf.batch_size
    flipud = conf.flipud

    for trx_ndx in range(n_trx):
        cur_trx = T[trx_ndx]
        # pre allocate results
        if first_frames[trx_ndx] > start_frame:
            cur_start = first_frames[trx_ndx]
        else:
            cur_start = start_frame

        if end_frames[trx_ndx] < end_frame:
            cur_end = end_frames[trx_ndx]
        else:
            cur_end = end_frame

        n_frames = cur_end - cur_start
        n_batches = int(math.ceil(float(n_frames) / bsize))
        all_f = np.zeros((bsize,) + conf.imsz + (conf.imgDim,))

        for curl in range(n_batches):
            ndx_start = curl * bsize + cur_start
            ndx_end = min(n_frames, (curl + 1) * bsize) + cur_start
            ppe = min(ndx_end - ndx_start, bsize)

            for ii in range(ppe):
                fnum = ndx_start + ii
                frame_in, cur_loc = multiResData.get_patch(
                    cap, fnum, conf, np.zeros([conf.n_classes, 2]),cur_trx=cur_trx,flipud=flipud)
                all_f[ii, ...] = frame_in

            base_locs = pred_fn(all_f)

            out_start = ndx_start - start_frame
            out_end = ndx_end - start_frame
            trx_fnum_start = ndx_start - first_frames[trx_ndx]
            base_locs_orig = convert_to_orig(base_locs, conf, cur_trx, trx_fnum_start, all_f, sz, ppe)
            pred_locs[out_start:out_end, trx_ndx, :, :] = base_locs_orig[:ppe, ...]
            if curl % 20 == 19:
                sys.stdout.write('.')
            if curl % 400 == 19:
                sys.stdout.write('\n')
                write_trk(pred_locs, ndx_end - start_frame)

        sys.stdout.write('\n')

    write_trk(pred_locs, end_frame - start_frame)
    cap.close()
    tf.reset_default_graph()
    return pred_locs

def classify_movie(conf, pred_fn, mov_file, out_file, trx_file=None, start_frame=0, end_frame=-1, skip_rate=1, trx_ids=[]):
    # classifies movies frame by frame instead of trx by trx.

    def write_trk(pred_locs_in,n_done):
        pred_locs = pred_locs_in.copy()
        pred_locs = pred_locs.transpose([2, 3, 0, 1])
        pred_locs = pred_locs[:,:,n_done,:]
        tgt = np.arange(pred_locs.shape[-1]) + 1
        if not conf.has_trx_file:
            pred_locs = pred_locs[...,0]
        ts_shape = pred_locs.shape[0:1] + pred_locs.shape[2:]
        ts = np.ones(ts_shape) * datetime2matlabdn()
        tag = np.zeros(ts.shape).astype('bool')
        tracked_shape = pred_locs.shape[2]
        tracked = np.zeros([1,tracked_shape])
        tracked[0,:] = np.array(n_done)+1
        hdf5storage.savemat(out_file,
                            {'pTrk': pred_locs, 'pTrkTS': ts, 'expname': mov_file, 'pTrkiTgt': tgt,
                             'pTrkTag': tag,'pTrkFrm':tracked}, appendmat=False, truncate_existing=True)

    cap = movies.Movie(mov_file)
    sz = (cap.get_height(), cap.get_width())
    n_frames = int(cap.get_n_frames())
    if conf.has_trx_file:
        T = sio.loadmat(trx_file)['trx'][0]
        n_trx = len(T)
        end_frames = np.array([x['endframe'][0, 0] for x in T])
        first_frames = np.array([x['firstframe'][0, 0] for x in T]) - 1  # for converting from 1 indexing to 0 indexing
        if len(trx_ids) == 0:
            trx_ids = range(n_trx)
    else:
        T = [None,]
        n_trx = 1
        end_frames = np.array([n_frames])
        first_frames = np.array([0])
        trx_ids = [0]

    if end_frame < 0:
        end_frame = end_frames.max()
    if end_frame > end_frames.max():
        end_frame = end_frames.max()
    if start_frame > end_frame:
        return None

    max_n_frames = end_frames.max() - first_frames.min()
    min_first_frame = first_frames.min()
    pred_locs = np.zeros([max_n_frames, n_trx, conf.n_classes, 2])
    pred_locs[:] = np.nan

    bsize = conf.batch_size
    flipud = conf.flipud
    all_f = np.zeros((bsize,) + conf.imsz + (conf.imgDim,))

    to_do_list = []
    for cur_f in range(start_frame,end_frame):
        for t in range(n_trx):
            if trx_ids.count(t) == 0:
                continue
            if (end_frames[t]> cur_f) and (first_frames[t]<=cur_f):
                to_do_list.append([cur_f,t])

    n_list = len(to_do_list)
    n_batches = int(math.ceil(float(n_list) / bsize))
    for cur_b in range(n_batches):
        cur_start = cur_b*bsize
        ppe = min(n_list - cur_start, bsize)
        for cur_t in range(ppe):
            cur_entry = to_do_list[cur_t + cur_start]
            trx_ndx = cur_entry[1]
            cur_trx = T[trx_ndx]
            cur_f = cur_entry[0]

            frame_in, cur_loc = multiResData.get_patch(
                cap, cur_f, conf, np.zeros([conf.n_classes, 2]),cur_trx=cur_trx,flipud=flipud)
            all_f[cur_t, ...] = frame_in

        base_locs = pred_fn(all_f)
        for cur_t in range(ppe):
            cur_entry = to_do_list[cur_t + cur_start]
            trx_ndx = cur_entry[1]
            cur_trx = T[trx_ndx]
            cur_f = cur_entry[0]
            trx_fnum_start = cur_f - first_frames[trx_ndx]
            base_locs_orig = convert_to_orig(base_locs[cur_t:cur_t+1,...], conf, cur_trx, trx_fnum_start, all_f, sz,1)
            pred_locs[cur_f-min_first_frame, trx_ndx, :, :] = base_locs_orig[0, ...]

        if cur_b%20==19:
            sys.stdout.write('.')
        if cur_b % 400 == 399:
            sys.stdout.write('\n')
            write_trk(pred_locs,range(start_frame,to_do_list[cur_start][0]))

    write_trk(pred_locs, range(start_frame,end_frame))
    cap.close()
    tf.reset_default_graph()
    return pred_locs


def classify_movie_unet(conf, mov_file, trx_file, out_file, start_frame=0, end_frame=-1, skip_rate=1, trx_ids = []):
    # classify movies using unet network.
    # this is the function that should be changed for different networks.
    # The main thing to do here is to define the pred_fn
    # which takes as input a batch of images and
    # returns the predicted locations.

    tf.reset_default_graph()
    self = PoseUNet.PoseUNet(conf)
    sess = self.init_net_meta(0, True)

    def pred_fn(all_f):
        # this is the function that is passed to classify_movie.
        # this should take in an array B x H x W x C of images, and
        # output an array of predicted locations.
        # predicted locations should be B x N x 2
        # PoseTools.get_pred_locs can be used to convert heatmaps into locations.

        bsize = conf.batch_size
        xs, _ = PoseTools.preprocess_ims(
            all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]), conf=self.conf,
            distort=False, scale=self.conf.unet_rescale)

        self.fd[self.ph['x']] = xs
        self.fd[self.ph['phase_train']] = False
        self.fd[self.ph['keep_prob']] = 1.
        pred = sess.run(self.pred, self.fd)
        base_locs = PoseTools.get_pred_locs(pred)
        base_locs = base_locs * conf.unet_rescale
        return base_locs

    classify_movie(conf, pred_fn, mov_file, out_file, trx_file, start_frame, end_frame, skip_rate, trx_ids)


def train(lblfile, nviews, name, view):
    if view is None:
        views = range(nviews)
    else:
        views = [view]

    for cur_view in views:
        conf = create_conf(lblfile, cur_view, name)
        conf.view = cur_view
        create_tfrecord(conf,False)
        tf.reset_default_graph()
        self = PoseUNet.PoseUNet(conf)
        self.train_unet(False, 1)

# def classify(lbl_file, n_views, name, mov_file, trx_file, out_file, start_frame, end_frame, skip_rate):
#     # print(mov_file)
#
#     for view in range(n_views):
#         conf = create_conf(lbl_file, view, name)
#         tf.reset_default_graph()
#         self = PoseUNet.PoseUNet(conf)
#         sess = self.init_net_meta(0, True)
#
#         cap = movies.Movie(mov_file[view], interactive=False)
#         n_frames = int(cap.get_n_frames())
#         height = int(cap.get_height())
#         width = int(cap.get_width())
#         cur_end_frame = end_frame
#         if end_frame < 0:
#             cur_end_frame = n_frames
#         cap.close()
#
#         if trx_file is not None:
#             pred_list = self.classify_movie_trx(mov_file[view], trx_file[view], sess, end_frame, start_frame)
#             temp_predScores = pred_list[1]
#             predScores = -1 * np.ones((n_frames,) + temp_predScores.shape[1:])
#             # predScores[start_frame:cur_end_frame, ...] = temp_predScores
#             # dummy predscores for now.
#
#             temp_pred_locs = pred_list
#             pred_locs = np.nan * np.ones((n_frames,) + temp_pred_locs.shape[1:])
#             pred_locs[start_frame:cur_end_frame, ...] = temp_pred_locs
#             pred_locs = pred_locs.transpose([2, 3, 0, 1])
#             tgt = np.arange(pred_locs.shape[-1]) + 1
#         else:
#             pred_list = self.classify_movie(mov_file[view], sess, end_frame, start_frame)
#
#             rescale = conf.unet_rescale
#             orig_crop_loc = conf.cropLoc[(height, width)]
#             crop_loc = [int(x / rescale) for x in orig_crop_loc]
#             end_pad = [int((height - conf.imsz[0]) / rescale) - crop_loc[0],
#                        int((width - conf.imsz[1]) / rescale) - crop_loc[1]]
#
#             pp = [(0, 0), (crop_loc[0], end_pad[0]), (crop_loc[1], end_pad[1]), (0, 0)]
#             temp_predScores = np.pad(pred_list[1], pp, mode='constant', constant_values=-1.)
#             predScores = np.zeros((n_frames,) + temp_predScores.shape[1:])
#             predScores[start_frame:cur_end_frame, ...] = temp_predScores
#
#             temp_pred_locs = pred_list[0]
#             temp_pred_locs[:, :, 0] += orig_crop_loc[1]
#             temp_pred_locs[:, :, 1] += orig_crop_loc[0]
#             pred_locs = np.nan * np.ones((n_frames,) + temp_pred_locs.shape[1:])
#             pred_locs[start_frame:cur_end_frame, ...] = temp_pred_locs
#             pred_locs = pred_locs.transpose([1, 2, 0])
#             tgt = np.arange(1) + 1
#
#         ts_shape = pred_locs.shape[0:1] + pred_locs.shape[2:]
#         ts = np.ones(ts_shape) * datetime2matlabdn()
#         tag = np.zeros(ts.shape).astype('bool')
#         hdf5storage.savemat(out_file + '_view_{}'.format(view) + '.trk',
#                             {'pTrk': pred_locs, 'pTrkTS': ts, 'expname': mov_file[view], 'pTrkiTgt': tgt,
#                              'pTrkTag': tag}, appendmat=False, truncate_existing=True)
#

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("lbl_file",
                        help="path to lbl file")
    parser.add_argument('-name',dest='name',help='Name for the run. Default - pose_unet', default='pose_unet')
    parser.add_argument('-view',dest='view',help='Run only for this view. If not specified, run for all views', default=None,type=int)
    subparsers = parser.add_subparsers(help='train or track', dest='sub_name')

    parser_train = subparsers.add_parser('train', help='Train the detector')
    # parser_train.add_argument('-cache',dest='cache_dir',
    #                           help='cache dir for training')

    parser_classify = subparsers.add_parser('track', help='Track a movie')
    parser_classify.add_argument("-mov", dest="mov",
                                 help="movie(s) to track", required=True, nargs='+')
    parser_classify.add_argument("-trx", dest="trx",
                                 help='trx file for above movie', default=None, nargs='*')
    parser_classify.add_argument('-start_frame', dest='start_frame', help='start tracking from this frame', type=int,
                                 default=0)
    parser_classify.add_argument('-end_frame', dest='end_frame', help='end frame for tracking', type=int, default=-1)
    parser_classify.add_argument('-skip_rate', dest='skip', help='frames to skip while tracking', default=1, type=int)
    parser_classify.add_argument('-out', dest='out_files', help='file to save tracking results to', required=True, nargs='+')
    parser_classify.add_argument('-trx_ids', dest='trx_ids', help='only track these animals', nargs='*',type=int,default=[])

    print(argv)
    args = parser.parse_args(argv)
    name = args.name

    lbl_file = args.lbl_file
    try:
        H = loadmat(lbl_file)
    except NotImplementedError:
        print('Label file is in v7.3 format. Loading using h5py')
        H = h5py.File(lbl_file)
    nviews = int(read_entry(H['cfg']['NumViews']))

    if args.sub_name == 'train':
        train(lbl_file, nviews, name, args.view)
    elif args.sub_name == 'track':
        assert len(args.mov) == len(args.out_files), 'Number of movie files and number of out files should be the same'

        if args.view is None:
            if args.trx is None:
                args.trx = [None] * nviews
            else:
                assert len(args.mov) == len(args.trx), 'Number of movie files should be same as the number of trx files'
            assert len(args.mov) == nviews, 'Number of movie files should be same as number of views'
            for view in range(nviews):
                conf = create_conf(lbl_file, view, name)
                classify_movie_unet(conf, args.mov[view], args.trx[view], args.out_files[view], args.start_frame, args.end_frame, args.skip, args.trx_ids)
        else:
            assert len(args.mov) == 1, 'Number of movie files should be one when view is speicified'
            conf = create_conf(lbl_file, args.view, name)
            classify_movie_unet(conf, args.mov, args.trx, args.out_files, args.start_frame,
                                args.end_frame, args.skip, args.trx_ids)


if __name__ == "__main__":
    main(sys.argv[1:])
