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
import copy
from matplotlib import cm
import matplotlib.pyplot as plt
import APT_interface
import datetime

split_type = 'easy'
# extra_str = '_normal_bnorm'
extra_str = '_my_bnorm'
imdim = 1

def get_cache_dir(conf, split, split_type):
    outdir = os.path.join(conf.cachedir, 'cv_split_{}'.format(split))
    if split_type is 'easy':
        outdir += '_easy'
    outdir += extra_str
    return outdir

def create_tfrecords():
    data_file = '/groups/branson/bransonlab/mayank/PoseTF/headTracking/trnDataSH_20180503_notable.mat'
    split_file = '/groups/branson/bransonlab/apt/experiments/data/trnSplits_20180509.mat'

    D = h5py.File(data_file,'r')
    S = APT_interface.loadmat(split_file)

    nims = D['IMain_crop2'].shape[1]
    movid = np.array(D['mov_id']).T - 1
    frame_num = np.array(D['frm']).T - 1
    if split_type is 'easy':
        split_arr = S['xvMain3Easy']
    else:
        split_arr = S['xvMain3Hard']


    for view in range(2):
        if view == 0:
            from stephenHeadConfig import sideconf as conf
        else:
            from stephenHeadConfig import conf as conf

        for split in range(3):
            outdir = get_cache_dir(conf, split, split_type)
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
    conf.imgDim = 1
    conf.cachedir = get_cache_dir(conf, split_num, split_type)
    self = PoseUNet.PoseUNet(conf)
    self.train_unet(False,0)


def submit_jobs():

    for view in range(2):
        for split_num in range(3):
            if view == 0:
                from stephenHeadConfig import sideconf as conf
            else:
                from stephenHeadConfig import conf as conf
            curconf = copy.deepcopy(conf)
            curconf.cachedir = get_cache_dir(conf, split_num, split_type)
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

def compile_results(split_type):
    all_dist = [[],[]]
    rims = [[],[]]
    rlocs = [[],[]]
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
            curconf.cachedir = get_cache_dir(conf, split_num, split_type)
            curconf.imgDim = 1
            tf.reset_default_graph()
            self = PoseUNet.PoseUNet(curconf)
            dist, ims, preds, predlocs, locs, info = self.classify_val(0)
            all_dist[view].append(dist)
        all_dist[view] = np.concatenate(all_dist[view],0)
        rims[view] = ims
        rlocs[view] = locs
    return all_dist, rims, rlocs

def create_result_images(split_type):
    dist, ims, locs = compile_results(split_type)
    in_locs = copy.deepcopy(locs)

    dstr = datetime.datetime.now().strftime('%Y%m%d')

    for view in range(2):
        perc = np.percentile(dist[view], [75, 90, 95, 98, 99, 99.5], axis=0)

        f, ax = plt.subplots()
        im = ims[view][0, :, :, 0]
        locs = in_locs[view][0, ...]

        ax.imshow(im) if im.ndim == 3 else ax.imshow(im, cmap='gray')
        # ax.scatter(locs[:,0],locs[:,1],s=20)
        cmap = cm.get_cmap('jet')
        rgba = cmap(np.linspace(0, 1, perc.shape[0]))
        for pndx in range(perc.shape[0]):
            for ndx in range(locs.shape[0]):
                ci = plt.Circle(locs[ndx, :], fill=False,
                                radius=perc[pndx, ndx], color=rgba[pndx, :])
                ax.add_artist(ci)
            plt.axis('off')
            f.savefig('/groups/branson/home/kabram/temp/stephen_unet_{}_view{}_results_{}.png'.format(dstr,view, pndx),
                      bbox_inches='tight', dpi=400)


def main(argv):
    split_num = int(argv[0])
    view = int(argv[1])
    print('split:{}, view:{}'.format(split_num,view))
    train(split_num, view)

if __name__ == "__main__":
    main(sys.argv[1:])
