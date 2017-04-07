import tensorflow as tf
from multiResData import *
import h5py
import pickle
import sys
import os
import PoseTrain
import multiResData
from stephenHeadConfig import sideconf as sideconf
from stephenHeadConfig import conf as frontconf


def main(argv):
    view = int(argv[1])
    curfold = int(argv[2])
    curgpu = argv[3]
    if len(argv)>=5:
        batch_size = int(argv[4])
    else:
        batch_size = -1

    trainfold(view=view,curfold=curfold,curgpu=curgpu,batch_size=batch_size)

def trainfold(view,curfold,curgpu,batch_size):
    if view == 0:
        conf = sideconf
    else:
        conf = frontconf

    if batch_size>0:
        conf.batch_size = batch_size

    createvaldata = False
    conf.cachedir = os.path.join(conf.cachedir,'cross_val_fly')

    if createvaldata:
        ##
        L  = h5py.File(conf.labelfile)
        localdirs,seldirs = findLocalDirs(conf)

        ##
        pts = L['labeledpos']
        nmov = len(localdirs)
        fly_num = np.zeros([nmov,])
        num_labels = np.zeros([nmov,])
        for ndx in range(nmov):
            f_str = re.search('fly_*(\d*)',localdirs[ndx])
            fly_num[ndx] = int(f_str.groups()[0])
            curpts = np.array(L[pts[0,ndx]])
            frames = np.where(np.invert(np.all(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
            num_labels[ndx] = len(frames)
        ##

        ufly,uidx = np.unique(fly_num,return_inverse=True)
        fly_labels = np.zeros([len(ufly)])
        for ndx in range(len(ufly)):
            fly_labels[ndx] = np.sum(num_labels[uidx==ndx])

        ##

        folds = 5
        lbls_fold = int(np.sum(num_labels)/folds)

        imbalance = True
        while imbalance:
            per_fold = np.zeros([folds])
            fly_fold = np.zeros(len(ufly))
            for ndx in range(len(ufly)):
                done = False
                curfold = np.random.randint(folds)
                while not done:
                    if per_fold[curfold]>lbls_fold:
                        curfold = (curfold+1)%folds
                    else:
                        fly_fold[ndx] = curfold
                        per_fold[curfold] += fly_labels[ndx]
                        done = True
            imbalance = (per_fold.max()-per_fold.min())>(lbls_fold/3)
        print per_fold

        ##

        for ndx in range(folds):
            curvaldatafilename = os.path.join(conf.cachedir,conf.valdatafilename + '_fold_{}'.format(ndx))
            fly_val = np.where(fly_fold==ndx)[0]
            isval = np.where(np.in1d(uidx,fly_val))[0].tolist()
            with open(curvaldatafilename,'w') as f:
                pickle.dump([isval,localdirs,seldirs],f)

    ##

    ext = '_fold_{}'.format(curfold)
    conf.valdatafilename = conf.valdatafilename + ext
    conf.trainfilename = conf.trainfilename + ext
    conf.valfilename = conf.valfilename + ext
    conf.fulltrainfilename += ext
    conf.baseoutname = conf.baseoutname + ext
    conf.mrfoutname += ext
    conf.fineoutname += ext
    conf.baseckptname += ext
    conf.mrfckptname += ext
    conf.fineckptname += ext
    conf.basedataname += ext
    conf.finedataname += ext
    conf.mrfdataname += ext

    _,localdirs,seldirs = multiResData.loadValdata(conf)
    for ndx,curl in enumerate(localdirs):
        if not os.path.exists(curl):
            print curl + ' {} doesnt exist!!!!'.format(ndx)
            return


    if not os.path.exists(os.path.join(conf.cachedir,conf.trainfilename+'.tfrecords')):
        multiResData.createTFRecordFromLbl(conf,True)
    os.environ['CUDA_VISIBLE_DEVICES'] = curgpu
    tf.reset_default_graph()
    self = PoseTrain.PoseTrain(conf)
    self.baseTrain(restore=True,trainPhase=True,trainType=0)
    tf.reset_default_graph()
    self.mrfTrain(restore=True,trainType=0)


if __name__ == "__main__":
    main(sys.argv)

def update_dirnames(conf):
##

    localdirs, seldirs = findLocalDirs(conf)

    newname = [False]*len(localdirs)
    for ndx in range(len(localdirs)):
        if localdirs[ndx].find('fly_450_to_452_26_9_1')>0:
            localdirs[ndx] = localdirs[ndx][:50] + 'fly_450_to_452_26_9_16norpAkirPONMNchr' + localdirs[ndx][72:]

    newname = [False]*len(localdirs)
    for ndx in range(len(localdirs)):
        if localdirs[ndx].find('fly_453_to_457_27_9_16')>0:
            localdirs[ndx] = localdirs[ndx][:50] + 'fly_453_to_457_27_9_16_norpAkirPONMNchr' + localdirs[ndx][72:]

    for ndx,curl in enumerate(localdirs):
        if not os.path.exists(curl):
            print curl + ' {} doesnt exist!!!!'.format(ndx)
    return localdirs


##
def createViewFiles(view):
    ##
    if view == 0:
        conf = sideconf
    else:
        conf = frontconf
    conf.cachedir = os.path.join(conf.cachedir,'cross_val_fly')

    basepath = '/groups/branson/bransonlab/mayank/stephenCV/'
    for curfold in range(5):
        ext = '_fold_{}'.format(curfold)
        curvaldatafilename = conf.valdatafilename + ext
        with open(os.path.join(conf.cachedir,curvaldatafilename), 'r') as f:
            [isval, localdirs, seldirs] = pickle.load(f)

        with open(basepath+'view{}vids_fold_{}.txt'.format(view+1,curfold),'w') as f:
            for ndx,curf in enumerate(localdirs):
                if not isval.count(ndx):
                    continue
                pp = curf.split('/')
                startat = -1
                for mndx in range(len(pp)-1,0,-1):
                    if re.match('^fly_*\d+$',pp[mndx]):
                        startat = mndx
                        break
                cc = '/'.join(pp[startat:])
                cc = basepath + cc
                if not os.path.exists(cc):
                    continue
                f.write(cc+'\n')
##
def createSideValData():
    from stephenHeadConfig import sideconf as sideconf
    localdirs, seldirs = findLocalDirs(sideconf)

    folds = 5
    for ndx in range(folds):
        curvaldatafilename = os.path.join(conf.cachedir, conf.valdatafilename + '_fold_{}'.format(ndx))
        with open(curvaldatafilename, 'r') as f:
            [isval, _a,_b] = pickle.load(f)
        curvaldatafilename = os.path.join(sideconf.cachedir, sideconf.valdatafilename + '_fold_{}'.format(ndx))
        with open(curvaldatafilename, 'w') as f:
            pickle.dump([isval, localdirs, seldirs], f)


##
# folds = 5
# allisval = []
# for ndx in range(1,folds):
#     curvaldatafilename = os.path.join(conf.cachedir, conf.valdatafilename + '_fold_{}'.format(ndx))
#     with open(curvaldatafilename, 'r') as f:
#         [isval, localdirs,seldirs] = pickle.load(f)
#     allisval += isval

##
# folds = 5
# for ndx in range(0,folds):
#     curvaldatafilename = os.path.join(conf.cachedir, conf.valdatafilename + '_fold_{}'.format(ndx))
#     with open(curvaldatafilename, 'r') as f:
#         [isval, _localdirs,seldirs] = pickle.load(f)
#     with open(curvaldatafilename, 'w') as f:
#         pickle.dump([isval, localdirs,seldirs] ,f)
