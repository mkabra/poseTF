##
import APT_interface
import h5py
import  tensorflow as tf
from multiResData import *
import localSetup
import os
import  PoseUNet
import  subprocess
import copy
import json

def create_tfrecords():
    data_file = '/groups/branson/bransonlab/mayank/PoseTF/headTracking/trnDataSH_20180503_notable.mat'
    split_file = '/groups/branson/bransonlab/apt/experiments/data/trnSplits_20180509.mat'

    D = h5py.File(data_file,'r')
    S = APT_interface.loadmat(split_file)

    ##

    nims = D['IMain_crop2'].shape[1]
    movid = np.array(D['mov_id']).T - 1
    frame_num = np.array(D['frm']).T - 1
    split_arr = S['xvMain3Hard']


    for view in range(2):
        if view == 0:
            from stephenHeadConfig import sideconf as conf
        else:
            from stephenHeadConfig import conf as conf

        for split in range(3):
            outdir = os.path.join(conf.cachedir, 'cv_split_{}'.format(split))
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            train_filename = os.path.join(outdir,conf.trainfilename)
            val_filename = os.path.join(outdir,conf.valfilename)
            env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')
            val_env = tf.python_io.TFRecordWriter(val_filename + '.tfrecords')
            splits = [[], []]
            all_locs = np.array(D['xyLblMain_crop2'])[view,:,:,:]
            for indx in range(nims):
                cur_im = np.array(D[D['IMain_crop2'][view,indx]]).T
                cur_locs = all_locs[...,indx].T
                mov_num = movid[indx]
                cur_frame_num = frame_num[indx]

                if split_arr[indx,split] == 1:
                    cur_env = val_env
                    splits[1].append([indx,cur_frame_num[0],0])
                else:
                    cur_env = env
                    splits[0].append([indx,cur_frame_num[0],0])


                im_raw = cur_im.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': int64_feature(cur_im.shape[0]),
                    'width': int64_feature(cur_im.shape[1]),
                    'depth': int64_feature(1),
                    'trx_ndx': int64_feature(0),
                    'locs': float_feature(cur_locs.flatten()),
                    'expndx':float_feature(mov_num),
                    'ts':float_feature(cur_frame_num[0]),
                    'image_raw':bytes_feature(im_raw)
                }))


                cur_env.write(example.SerializeToString())
            env.close()
            val_env.close()
            with open(os.path.join(outdir, 'splitdata.json'), 'w') as f:
                json.dump(splits, f)

    D.close()

def train(split_num, view):
    if view == 0:
        from stephenHeadConfig import sideconf as conf
        conf.imsz = (350,230)
    else:
        from stephenHeadConfig import conf as conf
        conf.imsz = (350,350)

    conf.batch_size = 4
    conf.unet_rescale = 1
    conf.cachedir = os.path.join(conf.cachedir, 'cv_split_{}'.format(split_num))
    self = PoseUNet.PoseUNet(conf)
    self.train_unet(True,0)


def submit_jobs():

    for view in range(2):
        for split_num in range(3):
            if view == 0:
                from stephenHeadConfig import sideconf as conf
            else:
                from stephenHeadConfig import conf as conf
            curconf = copy.deepcopy(conf)
            curconf.cachedir = os.path.join(curconf.cachedir, 'cv_split_{}'.format(split_num))
            sing_script = os.path.join(curconf.cachedir,'singularity_script.sh')
            sing_log = os.path.join(curconf.cachedir,'singularity_script.log')
            sing_err = os.path.join(curconf.cachedir,'singularity_script.err')
            with open(sing_script, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('. /opt/venv/bin/activate\n')
                f.write('cd /groups/branson/home/kabram/PycharmProjects/poseTF\n')
                f.write("if nvidia-smi | grep -q 'No devices were found'; then \n")
                f.write('{ echo "No GPU devices were found. quitting"; exit 1; }\n')
                f.write('fi\n')
                f.write('numCores2use={} \n'.format(1))
                f.write('python script_stephen_allen_data.py {} {}'.format(split_num,view))

            os.chmod(sing_script, 0755)

            cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {} -eo {} -n2 -gpu "num=1" -q gpu_any "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                sing_log, sing_err, sing_script)  # -n2 because SciComp says we need 2 slots for the RAM
            subprocess.call(cmd, shell=True)
            print('Submitted jobs for batch split:{} view:{}'.format(split_num,view))

def compile_results():
    all_dist = [[],[]]
    for view in range(2):
        for split_num in range(3):
            if view == 0:
                from stephenHeadConfig import sideconf as conf
                curconf = copy.deepcopy(conf)
                curconf.imsz = (350, 230)
            else:
                from stephenHeadConfig import conf as conf
                curconf = copy.deepcopy(conf)
                curconf.imsz = (350, 350)
            curconf.batch_size = 4
            curconf.unet_rescale = 1
            curconf.cachedir = os.path.join(curconf.cachedir, 'cv_split_{}'.format(split_num))
            tf.reset_default_graph()
            self = PoseUNet.PoseUNet(curconf)
            dist, ims, preds, predlocs, locs, info = self.classify_val(0)
            all_dist[view].append(dist)
        all_dist[view] = np.concatenate(all_dist[view],0)
    return all_dist

def main(argv):
    split_num = int(argv[0])
    view = int(argv[1])
    print('split:{}, view:{}'.format(split_num,view))
    train(split_num, view)

if __name__ == "__main__":
    main(sys.argv[1:])
