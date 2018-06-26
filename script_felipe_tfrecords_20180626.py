import PoseTools
import socket
import numpy as np
import os
import imageio
import tensorflow as tf
from multiResData import int64_feature, float_feature, bytes_feature

for dtype in ['train','val']:

    cachedir = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/'
    if dtype == 'train':
        ann_file = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/annotations/bee_keypoints_train.json'
        im_dir = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/train2017bee'
        out_filename = os.path.join(cachedir, 'train_TF.tfrecords')
    else:
        ann_file = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/annotations/bee_keypoints_val.json'
        im_dir = '/groups/branson/bransonlab/mayank/PoseTF/cache/felipe/val2017bee'
        out_filename = os.path.join(cachedir, 'val_TF.tfrecords')

    env =  tf.python_io.TFRecordWriter(out_filename)

    L = PoseTools.json_load(ann_file)
    ann_im_id = np.array([k['image_id'] for k in L['annotations']])
    bbox = np.array([k['bbox'] for k in L['annotations']] )
    locs = np.array([k['keypoints'] for k in L['annotations']] )

    n_labels = len(L['annotations'])
    locs = locs.reshape([n_labels,-1,2])
    bbox = bbox.reshape([n_labels,4,2])

    mid_pt = bbox.mean(axis=1)
    hsz = 250

    im_id = [k['id'] for k in L['images']]
    im_name = [k['file_name'] for k in L['images']]

    for ndx in range(n_labels):
        im_ndx = np.where(im_id==ann_im_id[ndx])[0]
        cur_im = os.path.join(im_dir,im_name[im_ndx])
        im = imageio.imread(cur_im)

        cur_mid_x,cur_mid_y = mid_pt[ndx,:]
        cur_patch = im[cur_mid_y-hsz:cur_mid_y+hsz,cur_mid_x-hsz:cur_mid_x+hsz,:]
        cur_locs = locs[ndx,:,:]-mid_pt[ndx:ndx+1,:] + hsz

        image_raw = cur_patch.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(cur_patch.shape[0]),
            'width': int64_feature(cur_patch.shape[1]),
            'depth': int64_feature(cur_patch.shape[2]),
            'trx_ndx': int64_feature(0),
            'locs': float_feature(cur_locs.flatten()),
            'expndx': float_feature(ndx),
            'ts': float_feature(ndx),
            'image_raw': bytes_feature(image_raw)}))
        env.write(example.SerializeToString())

    env.close()