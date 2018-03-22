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
# import hdf5storage

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("lbl_file",
                        help="path to lbl file")
    subparsers = parser.add_subparsers(help='train or track')

    parser_train = subparsers.add_parser('train',help='Train the detector')
    parser_train.add_argument('-cache',dest='cache_dir',
                              help='cache dir for training')

    parser_classify = subparsers.add_parser('track',help='Track a movie')
    parser_classify.add_argument("-mov", dest="mov",
                        help="movie to track",required=True)
    parser_classify.add_argument("-trx", dest="trx",
                        help='trx file for above movie')
    parser_classify.add_argument('-start_frame',dest='start_frame', help='start tracking from this frame',type=int,default=0)
    parser_classify.add_argument('-end_frame',dest='end_frame', help='end frame for tracking',type=int,default=-1)
    parser_classify.add_argument('-skip_rate', dest='skip',help='frames to skip while tracking',default=1,type=int)
    parser_classify.add_argument('-out_file', dest='out_file',help='file to save tracking results to',required=True)

    args = parser.parse_args()


if __name__ == "__main__":
    main(sys.argv[1:])

