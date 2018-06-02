from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[13]:

#!/usr/bin/python
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
import hdf5storage

default_net_name = 'pose_unet_full_20180302'

def main(argv):

#    defaulttrackerpath = "/groups/branson/home/bransonk/tracking/code/poseTF/matlab/compute3Dfrom2D/for_redistribution_files_only/run_compute3Dfrom2D.sh"
    defaulttrackerpath = "/groups/branson/bransonlab/mayank/PoseTF/matlab/compiled/run_compute3Dfrom2D_compiled.sh"
#    defaultmcrpath = "/groups/branson/bransonlab/projects/olympiad/MCR/v91"
    defaultmcrpath = "/groups/branson/bransonlab/mayank/MCR/v92"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s",dest="sfilename",
                      help="text file with list of side view videos",
                      required=True)
    parser.add_argument("-f",dest="ffilename",
                      help="text file with list of front view videos. The list of side view videos and front view videos should match up",
                      required=True)
    parser.add_argument("-d",dest="dltfilename",
                      help="text file with list of DLTs, one per fly as 'flynum,/path/to/dltfile'",
                      required=True)
    parser.add_argument("-net",dest="net_name",
                      help="Name of the net to use for tracking",
                      default=default_net_name)
    parser.add_argument("-o",dest="outdir",
                      help="temporary output directory to store intermediate computations",
                      required=True)
    parser.add_argument("-r",dest="redo",
                      help="if specified will recompute everything",
                      action="store_true")
    parser.add_argument("-rt",dest="redo_tracking",
                      help="if specified will only recompute tracking",
                      action="store_true")
    parser.add_argument("-gpu",dest='gpunum',type=int,
                        help="GPU to use [optional]")
    parser.add_argument("-makemovie",dest='makemovie',
                        help="if specified will make results movie",action="store_true")
    parser.add_argument("-trackerpath",dest='trackerpath',
                        help="Absolute path to the compiled MATLAB tracker script run_compute3Dfrom2D.sh",
                        default=defaulttrackerpath)
    parser.add_argument("-mcrpath",dest='mcrpath',
                        help="Absolute path to MCR",
                        default=defaultmcrpath)
    parser.add_argument("-ncores",dest="ncores",
                        help="Number of cores to assign to each MATLAB tracker job", type=int,
                        default=1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-only_detect",dest='detect',action="store_true",
                        help="Do only the detection part of tracking which requires GPU")
    group.add_argument("-only_track",dest='track',action="store_true",
                        help="Do only the tracking part of the tracking which requires MATLAB")

    args = parser.parse_args(argv)
    if args.redo is None:
        args.redo = False
    if args.redo_tracking is None:
        args.redo_tracking= False

    if args.detect is False and args.track is False: 
        args.detect = True
        args.track = True
    
    args.outdir = os.path.abspath(args.outdir)
    
    with open(args.sfilename, "r") as text_file:
        smovies = text_file.readlines()
    smovies = [x.rstrip() for x in smovies]
    with open(args.ffilename, "r") as text_file:
        fmovies = text_file.readlines()
    fmovies = [x.rstrip() for x in fmovies]

    print(smovies)
    print(fmovies)
    print(len(smovies))
    print(len(fmovies))

    if args.track:
        if len(smovies) != len(fmovies):
            print("Side and front movies must match")
            raise exit(0)

        # read in dltfile
        dltdict = {}
        f = open(args.dltfilename,'r')
        for l in f:
            lparts = l.split(',')
            if len(lparts) != 2:
                print("Error splitting dlt file line %s into two parts"%l)
                raise exit(0)
            dltdict[float(lparts[0])] = lparts[1].strip()
        f.close()
        
        # compiled matlab command
        matscript = args.trackerpath + " " + args.mcrpath

    if args.detect:
        import numpy as np
        import tensorflow as tf
        from scipy import io
        from cvc import cvc
        import localSetup
        import PoseTools
        import multiResData
        import cv2
        import PoseUNet

        for ff in smovies+fmovies:
            if not os.path.isfile(ff):
                print("Movie %s not found"%(ff))
                raise exit(0)
        if args.gpunum is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    for view in range(2): # 0 for front and 1 for side
        if args.detect:
            tf.reset_default_graph() 
        if view ==1:
            from stephenHeadConfig import sideconf as conf
            extrastr = '_side'
            valmovies = smovies
            confname = 'sideconf'
        else:
            # For FRONT
            from stephenHeadConfig import conf as conf
            extrastr = '_front'
            valmovies = fmovies
            confname = 'conf'

        # for ndx in range(len(valmovies)):
        #     mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
        #     oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))
        #     pname = os.path.join(args.outdir , oname + extrastr)
        #     print(oname)
        #
        #     if args.detect and os.path.isfile(valmovies[ndx]) and \
        #                    (args.redo or not os.path.isfile(pname + '.mat')):
        #
        #         detect_cmd = 'python classifyMovie.py stephenHeadConfig {} {} {}'.format(confname, net_name, valmovies[ndx], pname+'.mat')

        # conf.batch_size = 1

        if args.detect:        
            for try_num in range(4):
                try:
                    self = PoseUNet.PoseUNet(conf, args.net_name)
                    sess = self.init_net_meta(0,True)
                    break
                except tf.python.framework.errors_impl.InvalidArgumentError:
                    tf.reset_default_graph()
                    print('Loading the net failed, retrying')
                    if try_num is 3:
                        raise ValueError('Couldnt load the network after 4 tries')

        for ndx in range(len(valmovies)):
            mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
            oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))
            pname = os.path.join(args.outdir , oname + extrastr)

            print(oname)

            # detect
            if args.detect and os.path.isfile(valmovies[ndx]) and \
               (args.redo or not os.path.isfile(pname + '.mat')):

                try:
                    predList = self.classify_movie(valmovies[ndx], sess, flipud=False)
                except KeyError:
                    continue
                # if args.makemovie:
                #     PoseTools.create_pred_movie(conf, predList, valmovies[ndx], pname + '.avi', outtype)

                cap = cv2.VideoCapture(valmovies[ndx])
                height = int(cap.get(cvc.FRAME_HEIGHT))
                width = int(cap.get(cvc.FRAME_WIDTH))
                rescale = conf.unet_rescale
                orig_crop_loc = conf.cropLoc[(height,width)]
                crop_loc = [int(x/rescale) for x in orig_crop_loc]
                end_pad = [int((height-conf.imsz[0])/rescale)-crop_loc[0],int((width-conf.imsz[1])/rescale)-crop_loc[1]]
#                crop_loc = [old_div(x,4) for x in orig_crop_loc]
#                end_pad = [old_div(height,4)-crop_loc[0]-old_div(conf.imsz[0],4),old_div(width,4)-crop_loc[1]-old_div(conf.imsz[1],4)]
                pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0)]
                predScores = np.pad(predList[1],pp,mode='constant',constant_values=-1.)

                predLocs = predList[0]
                predLocs[:,:,0] += orig_crop_loc[1]
                predLocs[:,:,1] += orig_crop_loc[0]

#io.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores,'expname':valmovies[ndx]})
                hdf5storage.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores,'expname':valmovies[ndx]},appendmat=False,truncate_existing=True,gzip_compression_level=0)
#                with h5py.File(pname+'.mat','w') as f:
#                    f.create_dataset('locs',data=predLocs)
#                    f.create_dataset('scores',data=predScores)
#                    f.create_dataset('expname',data=valmovies[ndx])
                del predScores, predLocs

                print('Detecting:%s'%oname)

            # track
            if args.track and view == 1:

                oname_side = re.sub('!','__',conf.getexpname(smovies[ndx]))
                oname_front = re.sub('!','__',conf.getexpname(fmovies[ndx]))
                pname_side = os.path.join(args.outdir , oname_side + '_side.mat')
                pname_front = os.path.join(args.outdir , oname_front + '_front.mat')
                # 3d trajectories
                basename_front,_ = os.path.splitext(fmovies[ndx])
                basename_side,_ = os.path.splitext(smovies[ndx])
                savefile = basename_side+'_3Dres.mat'
                #savefile = os.path.join(args.outdir , oname_side + '_3Dres.mat')
                trkfile_front = basename_front+'.trk'
                trkfile_side = basename_side+'.trk'

                redo_tracking = args.redo or args.redo_tracking
                if os.path.isfile(savefile) and os.path.isfile(trkfile_front) and \
                   os.path.isfile(trkfile_side) and not redo_tracking:
                    print("%s, %s, and %s exist, skipping tracking"%(savefile,trkfile_front,trkfile_side))
                    continue

                try:
                    flynum = int(conf.getflynum(smovies[ndx]))
                except AttributeError:
                    print('Could not find the fly number from movie name')
                    print('{} isnt in standard format'.format(smovies[ndx]))
                    continue
                #print "Parsed fly number as %d"%flynum
                kinematfile = os.path.abspath(dltdict[flynum])

                jobid = oname_side

                scriptfile = os.path.join(args.outdir , jobid + '_track.sh')
                logfile = os.path.join(args.outdir , jobid + '_track.log')
                errfile = os.path.join(args.outdir , jobid + '_track.err')


                #print "matscript = " + matscript
                #print "pname_front = " + pname_front
                #print "pname_side = " + pname_side
                #print "kinematfile = " + kinematfile
                
                # make script to be qsubbed
                scriptf = open(scriptfile,'w')
                scriptf.write('if [ -d %s ]\n'%args.outdir)
                scriptf.write('  then export MCR_CACHE_ROOT=%s/mcrcache%s\n'%(args.outdir,jobid))
                scriptf.write('fi\n')
                scriptf.write('%s "%s" "%s" "%s" "%s" "%s" "%s"\n'%(matscript,savefile,pname_front,pname_side,kinematfile,trkfile_front,trkfile_side))
                scriptf.write('chmod g+w {}\n'.format(savefile))
                scriptf.write('chmod g+w {}\n'.format(trkfile_front))
                scriptf.write('chmod g+w {}\n'.format(trkfile_side))
                scriptf.close()
                os.chmod(scriptfile,stat.S_IRUSR|stat.S_IRGRP|stat.S_IWUSR|stat.S_IWGRP|stat.S_IXUSR|stat.S_IXGRP)

#                cmd = "ssh login1 'source /etc/profile; qsub -pe batch %d -N %s -j y -b y -o '%s' -cwd '\"%s\"''"%(args.ncores,jobid,logfile,scriptfile)
                cmd = "ssh login2 'source /etc/profile; bsub -n %d -J %s -oo '%s' -eo '%s' -cwd . '\"%s\"''"%(args.ncores,jobid,logfile,errfile,scriptfile)
                print(cmd)
                call(cmd,shell=True)
                
if __name__ == "__main__":
   main(sys.argv[1:])

