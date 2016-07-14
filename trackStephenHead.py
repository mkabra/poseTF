
# coding: utf-8

# In[13]:

#!/usr/bin/python
import os
import numpy as np
import re
import glob
import sys
import tensorflow as tf
from scipy import io
from cvc import cvc
import localSetup
import PoseTools
import multiResData
import argparse
import cv2


def main(argv):
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-s",dest="sfilename",
                      help="text file with list of side view videos",
                      required=True)
    parser.add_argument("-f",dest="ffilename",
                      help="text file with list of front view videos",
                      required=True)
    parser.add_argument("-d",dest="dltfilename",
                      help="text file with list of DLTs one per fly as 'flynum,/path/to/dltfile",
                      required=True)
    parser.add_argument("-o",dest="outdir",
                      help="temporary output directory to store intermediate computations",
                      required=True)
    parser.add_argument("-r",dest="redo",
                      help="force recompute tracks for all videos",
                      action="store_true")
    parser.add_argument("-gpu",dest='gpunum',type=int,
                        help="GPU to use")
    parser.add_argument("-makemovie",dest='makemovie',
                        help="make results movie",action="store_true")
#     parser.add_argument("-conf",dest='conf',
#                         help="Config files")
#     parser.add_argument("-nviews",dest='nviews',type=int,
#                         help="make results movie")

    args = parser.parse_args()
    if args.redo is None:
        args.redo = False
    
    with open(args.sfilename, "r") as text_file:
        smovies = text_file.readlines()
    smovies = [x.rstrip() for x in smovies]
    with open(args.ffilename, "r") as text_file:
        fmovies = text_file.readlines()
    fmovies = [x.rstrip() for x in fmovies]

    print smovies
    print fmovies
    print len(smovies)
    print len(fmovies)
    
    for ff in smovies+fmovies:
        if not os.path.isfile(ff):
            print "Movie not found %s"%(ff)
            raise exit(0)
    if args.gpunum is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    for view in range(2): # 0 for front and 1 for side
        tf.reset_default_graph() 
        if view ==1:

            from stephenHeadConfig import sideconf as conf
            conf.useMRF = False
            outtype = 1
            extrastr = '_side'
            valmovies = smovies    
        else:
            # For FRONT
            from stephenHeadConfig import conf as conf
            conf.useMRF = True
            outtype = 2
            extrastr = '_front'
            valmovies = fmovies    

        # conf.batch_size = 1

        self = PoseTools.createNetwork(conf,outtype)
        sess = tf.InteractiveSession()
        PoseTools.initNetwork(self,sess,outtype)


        for ndx in range(len(valmovies)):
            mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
            oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))
        #     pname = '/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/' + oname + extrastr
            pname = os.path.join(args.outdir , oname + extrastr)
            print oname
            if os.path.isfile(pname + '.mat') and not args.redo:
                continue


            if not os.path.isfile(valmovies[ndx]):
                continue

            predList = PoseTools.classifyMovie(conf,valmovies[ndx],outtype,self,sess)
            
            if args.makemovie:
                PoseTools.createPredMovie(conf,predList,valmovies[ndx],pname + '.avi',outtype)

            cap = cv2.VideoCapture(valmovies[ndx])
            height = int(cap.get(cvc.FRAME_HEIGHT))
            width = int(cap.get(cvc.FRAME_WIDTH))
            orig_crop_loc = conf.cropLoc[(height,width)]
            crop_loc = [x/4 for x in orig_crop_loc] 
            end_pad = [height/4-crop_loc[0]-conf.imsz[0]/4,width/4-crop_loc[1]-conf.imsz[1]/4]
            pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0),(0,0)]
            predScores = np.pad(predList[1],pp,mode='constant',constant_values=-1.)

            predLocs = predList[0]
            predLocs[:,:,:,0] += orig_crop_loc[1]
            predLocs[:,:,:,1] += orig_crop_loc[0]

            io.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores[...,0],'expname':valmovies[ndx]})
            print 'Done:%s'%oname

    script_path = os.path.realpath(__file__)
    [script_dir,script_name] = os.path.split(script_path)
    matdir = os.path.join(script_dir,'matlab')
    matlab_cmd = "addpath %s; GMMTrack(''%s'',''%s'',''%s'',''%s'',%d);" %(matdir,args.ffilename,
                                                          args.sfilename,args.dltfilename,
                                                          args.outdir,args.redo)
    matlab_cmd = "matlab -nodesktop -nosplash -r '%s' " % matlab_cmd
    print 'Executing matlab command:%s'%matlab_cmd
    sys.exec(matlab_cmd)

if __name__ == "__main__":
   main(sys.argv[1:])

