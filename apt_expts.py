import  matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
import os
import APT_interface as apt
import h5py
import subprocess
import yaml
import pickle

methods = ['unet','leap','deeplabcut','openpose']
out_dir = '/groups/branson/bransonlab/mayank/apt_expts/'
nsplits = 3
openpose_dir = '/groups/branson/bransonlab/mayank/apt_expts/open_pose'
deepcut_dir = '/groups/branson/bransonlab/mayank/apt_expts/deepcut'
leap_dir = '/groups/branson/bransonlab/mayank/apt_expts/leap'
unet_dir = '/groups/branson/home/kabram/PycharmProjects/poseTF'
deepcut_default_cfg = '/groups/branson/bransonlab/mayank/PoseTF/cache/apt_interface/multitarget_bubble_view0/test_deepcut/pose_cfg.yaml'

def create_deepcut_cfg(conf):
    with open(deepcut_default_cfg,'r') as f:
        default_cfg = yaml.load(f)
    default_cfg['dataset'] = os.path.join(conf.cachedir, 'train_data.p')
    default_cfg['all_joints'] = [[i] for i in range(conf.n_classes)]
    default_cfg['all_joints_names'] = ['part_{}'.format(i) for i in range(conf.n_classes)]
    with open(os.path.join(conf.cachedir, 'pose_cfg.yaml'), 'w') as f:
        yaml.dump(default_cfg, f)


def check_db(curm, conf):
    check_file = os.path.join(conf.cachedir, 'test_train_db.p')
    apt.check_train_db(curm, conf, check_file)
    with open(check_file, 'r') as f:
        F = pickle.load(f)
    out_dir = os.path.join(conf.cachedir, 'train_check')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if F['ims'].max < 2:
        F['ims'] = F['ims']*255

    for ndx in range(F['ims'].shape[0]):
        plt.close('all')
        f = plt.figure()
        if F['ims'].shape[-1] == 1:
            plt.imshow(F['ims'][ndx,:,:,0].astype('uint8'),'gray')
        else:
            plt.imshow(F['ims'][ndx,:,:,:].astype('uint8'))
        plt.scatter(F['locs'][ndx,:,0],F['locs'][ndx,:,1])
        plt.savefig(os.path.join(out_dir,'train_{}.jpg'.format(ndx)))


def create_db(args):
    H = h5py.File(args.lbl_file,'r')
    nviews = int(apt.read_entry(H['cfg']['NumViews']))
    all_nets = args.nets

    for view in range(nviews):
        conf = apt.create_conf(args.lbl_file, view, args.name)
        for curm in all_nets:

            if args.split_type is None:

                conf.cachedir = os.path.join(out_dir,args.name,'common','{}_view_{}'.format(curm,view),'full')
                if not os.path.exists(conf.cachedir):
                    os.makedirs(conf.cachedir)
                if curm == 'unet' or curm == 'openpose':
                    apt.create_tfrecord(conf, False)
                elif curm == 'leap':
                    apt.create_leap_db(conf, False)
                elif curm == 'deepcut':
                    apt.create_deepcut_db(conf, False)
                    create_deepcut_cfg(conf)
                else:
                    raise ValueError('Undefined net type: {}'.format(curm))

                check_db(curm, conf)
            else:

                conf.cachedir = os.path.join(out_dir,args.name,'common','{}_view_{}'.format(curm,view))
                conf.splitType = args.split_type
                train_info, val_info, split_files = apt.create_cv_split_files(conf, nsplits)

                for cur_split in range(nsplits):
                    conf.cachedir = os.path.join(out_dir, args.name, '{}_view_{}'.format(curm,view), 'cv_{}'.format(cur_split))
                    conf.splitType = 'predefined'
                    split_file = split_files[cur_split]
                    if curm == 'unet' or curm == 'openpose':
                        apt.create_tfrecord(conf, True, split_file)
                    elif curm == 'leap':
                        apt.create_leap_db(conf, True, split_file)
                    elif curm == 'deepcut':
                        apt.create_deepcut_db(conf, True, split_file)
                        create_deepcut_cfg(conf)
                    else:
                        raise ValueError('Undefined net type: {}'.format(curm))
                    check_db(curm, conf)

        base_dir = os.path.join(out_dir, args.name, 'common')
        their_dir = os.path.join(out_dir, args.name, 'theirs')
        our_dir = os.path.join(out_dir, args.name, 'ours')
        our_default_dir = os.path.join(out_dir, args.name, 'ours_default')
        cmd = 'cp -rs {} {}'.format(base_dir, their_dir)
        os.system(cmd)
        cmd = 'cp -rs {} {}'.format(base_dir, our_dir)
        os.system(cmd)
        cmd = 'cp -rs {} {}'.format(base_dir, our_default_dir)
        os.system(cmd)


def train_theirs(args):
    H = h5py.File(args.lbl_file,'r')
    nviews = int(apt.read_entry(H['cfg']['NumViews']))
    all_nets = args.nets

    for view in range(nviews):
        conf = apt.create_conf(args.lbl_file, view, args.name)
        for curm in all_nets:

            if args.split_type is None:

                cachedir = os.path.join(out_dir,args.name,'theirs','{}_view_{}'.format(curm,view),'full')
                singularity_script = os.path.join(conf.cachedir,'singularity.sh')
                singularity_logfile = os.path.join(conf.cachedir,'singularity.log')
                f = open(singularity_script, 'w')
                f.write('#!/bin/bash\n')
                f.write('. /opt/venv/bin/activate\n')

                args.skip_db = True
                if curm == 'unet':
                    f.write('cd {}\n'.format(unet_dir))
                    cmd = 'APT_interface.py {} -view {} -cache {} -type unet -train '.format(args.lbl_file, view, cachedir)
                    f.write('python {}'.format(cmd))
                elif curm == 'openpose':
                    f.write('cd {}\n'.format(openpose_dir))
                    cmd = 'train_pose.py {} {}'.format(args.lbl_file, cachedir)
                    f.write('python {}'.format(cmd))
                elif curm == 'leap':
                    f.write('cd {}\n'.format(leap_dir))
                    cmd = 'training_MK.py {}'.format(cachedir)
                    f.write('python {}'.format(cmd))
                elif curm == 'deepcut':
                    f.write('cd {}\n'.format(cachedir))
                    cmd = os.path.join(deepcut_dir,'pose-tensorflow','train.py')
                    f.write('python {}'.format(cmd))
                else:
                    raise ValueError('Undefined net type: {}'.format(curm))

                f.close()
                os.chmod(singularity_script, 0755)
                cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {}  -n4 -gpu "num=1" -q gpu_any "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                    singularity_logfile, singularity_script)  # -n4 because we use 4 preprocessing threads
                subprocess.call(cmd, shell=True)
                print('Submitted job: {}'.format(cmd))

            else:

                conf.cachedir = os.path.join(out_dir,args.name,'theirs','{}_view_{}'.format(curm,view))
                conf.splitType = args.split_type
                train_info, val_info, split_files = apt.create_cv_split_files(conf, nsplits)

                for cur_split in range(nsplits):
                    conf.cachedir = os.path.join(out_dir, args.name, '{}_view_{}'.format(curm,view), 'cv_{}'.format(cur_split))
                    conf.splitType = 'predefined'
                    split_file = split_files[cur_split]
                    if curm == 'unet' or curm == 'openpose':
                        apt.create_tfrecord(conf, True, split_file)
                    elif curm == 'leap':
                        apt.create_leap_db(conf, True, split_file)
                    elif curm == 'deepcut':
                        apt.create_deepcut_db(conf, True, split_file)
                    else:
                        raise ValueError('Undefined net type: {}'.format(curm))


def train_ours(args):
    H = h5py.File(args.lbl_file,'r')
    nviews = int(apt.read_entry(H['cfg']['NumViews']))
    dir_name = 'ours_default'

    if len(args.nets) == 0:
        all_nets = methods
    else:
        all_nets = args.nets

    for view in range(nviews):
        conf = apt.create_conf(args.lbl_file, view, args.name)
        for curm in all_nets:

            if args.split_type is None:

                cachedir = os.path.join(out_dir,args.name,dir_name,'{}_view_{}'.format(curm,view),'full')
                singularity_script = os.path.join(conf.cachedir,'singularity.sh')
                singularity_logfile = os.path.join(conf.cachedir,'singularity.log')
                f = open(singularity_script, 'w')
                f.write('#!/bin/bash\n')
                f.write('. /opt/venv/bin/activate\n')

                f.write('cd {}\n'.format(unet_dir))
                cmd = 'APT_interface.py {} -view {} -cache {} -type {} -train '.format(args.lbl_file, view, cachedir, curm)
                if args.whose == 'ours_default':
                    cmd += ' -use_defaults'
                f.write('python {}'.format(cmd))
                f.close()
                os.chmod(singularity_script, 0755)
                cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {}  -n4 -gpu "num=1" -q gpu_any "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                    singularity_logfile, singularity_script)  # -n4 because we use 4 preprocessing threads
                subprocess.call(cmd, shell=True)
                print('Submitted job: {}'.format(cmd))

            else:

                for cur_split in range(nsplits):
                    conf.cachedir = os.path.join(out_dir, args.name, '{}_view_{}'.format(curm,view), 'cv_{}'.format(cur_split))
                    singularity_script = os.path.join(conf.cachedir,'singularity.sh')
                    singularity_logfile = os.path.join(conf.cachedir,'singularity.log')
                    f = open(singularity_script, 'w')
                    f.write('#!/bin/bash\n')
                    f.write('. /opt/venv/bin/activate\n')

                    f.write('cd {}\n'.format(unet_dir))
                    cmd = 'APT_interface.py {} -view {} -cache {} -type {} -train '.format(args.lbl_file, view, cachedir, curm)
                    if args.whose == 'ours_default':
                        cmd += ' -use_defaults'
                    f.write('python {}'.format(cmd))
                    f.close()
                    os.chmod(singularity_script, 0755)
                    cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {}  -n4 -gpu "num=1" -q gpu_any "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                        singularity_logfile, singularity_script)  # -n4 because we use 4 preprocessing threads
                    subprocess.call(cmd, shell=True)
                    print('Submitted job: {}'.format(cmd))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', dest='type', help='Create DB or train',
                        required=True, choices=['train','create_db'])
    parser.add_argument("-lbl_file",
                        help="path to lbl file", required=True)
    parser.add_argument('-name', dest='name', help='Name for the setup',
                        required=True)
    parser.add_argument('-split_type', dest='split_type',
                        help='Type of split for CV. If not defined not CV is done', default=None)
    parser.add_argument('-whose', dest='whose',
                        help='Use their or our code', default=None, choices=['theirs','ours','our_default'])
    parser.add_argument('-nets', dest='nets', help='Type of nets to run on. Options are unet, openpose, deeplabcut and leap. If not specified run on all nets',
            default=[], nargs='*')
    args = parser.parse_args(argv)
    if len(args.nets) == 0:
        args.nets = methods
    if args.type == 'create_db':
        create_db(args)
    elif args.type == 'train':
        if args.whose == 'theirs':
            train_theirs(args)
        else:
            train_ours(args)


if __name__ == "__main__":
    main(sys.argv[1:])
