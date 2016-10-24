
# coding: utf-8

# In[2]:

import tensorflow as tf
import os,sys
import caffe
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import cv2
import tempfile
import copy
import re

from batch_norm import batch_norm_2D
import myutils
import PoseTools
import localSetup
import operator
import copy


# In[ ]:

def poseEvalNet(lin,locs,conf,trainPhase,dropout):

    lin_sz = tf.Tensor.get_shape(lin).as_list()
    lin_numel = reduce(operator.mul, lin_sz[1:], 1)
    lin_re = tf.reshape(lin,[-1,lin_numel])
    lin_re = tf.nn.dropout(lin_re,dropout)
    with tf.variable_scope('lin_fc'):
        weights = tf.get_variable("weights", [lin_numel, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        lin_fc = tf.nn.relu(batch_norm_2D(tf.matmul(lin_re,weights)+biases,trainPhase))

        
    loc_sz = tf.Tensor.get_shape(locs).as_list()
    loc_numel = reduce(operator.mul, loc_sz[1:], 1)
    loc_re = tf.reshape(locs,[-1,loc_numel])
    with tf.variable_scope('loc_fc'):
        weights = tf.get_variable("weights", [loc_numel, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        loc_fc = tf.nn.relu(batch_norm_2D(tf.matmul(loc_re,weights)+biases,trainPhase))
        
    joint_fc = tf.concat(1,[lin_fc,loc_fc])
    
    with tf.variable_scope('fc1'):
        weights = tf.get_variable("weights", [conf.nfcfilt*2, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        joint_fc1 = tf.nn.relu(batch_norm_2D(tf.matmul(joint_fc,weights)+biases,trainPhase))

    with tf.variable_scope('fc2'):
        weights = tf.get_variable("weights", [conf.nfcfilt, conf.nfcfilt],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", conf.nfcfilt,
            initializer=tf.constant_initializer(0))
        
        joint_fc2 = tf.nn.relu(batch_norm_2D(tf.matmul(joint_fc1,weights)+biases,trainPhase))
        
    with tf.variable_scope('out'):
        weights = tf.get_variable("weights", [conf.nfcfilt, 2],
            initializer=tf.random_normal_initializer(stddev=0.005))
        biases = tf.get_variable("biases", 2,
            initializer=tf.constant_initializer(0))
        
        out = tf.matmul(joint_fc2,weights)+biases
        
    layer_dict = {'lin_fc':lin_fc,
                  'loc_fc':loc_fc,
                  'joint_fc1':joint_fc1,
                  'joint_fc2':joint_fc2,
                  'out':out
                 }
    return out,layer_dict
    
    


# In[ ]:

def createEvalPH(conf):
    lin = tf.placeholder(tf.float32,[None,conf.n_classes,conf.nfcfilt])
    locs = tf.placeholder(tf.float32,[None,conf.n_classes,2])
    learning_rate_ph = tf.placeholder(tf.float32,shape=[])
    y = tf.placeholder(tf.float32,[None,2])
    phase_train = tf.placeholder(tf.bool, name='phase_train')                 
    dropout = tf.placeholder(tf.float32, shape=[])                 
    phDict = {'lin':lin,'locs':locs,'learning_rate':learning_rate_ph,
              'y':y,'phase_train':phase_train,'dropout':dropout}
    return phDict


# In[ ]:

def createFeedDict(phDict):
    feed_dict = {phDict['lin']:[],
                 phDict['locs']:[],
                 phDict['y']:[],
                 phDict['learning_rate']:1.,
                 phDict['phase_train']:False,
                 phDict['dropout']:1.
                }
    return feed_dict


# In[ ]:

def createEvalSaver(conf):
    evalSaver = tf.train.Saver(var_list = PoseTools.getvars('poseEval'),max_to_keep=conf.maxckpt)
    return evalSaver


# In[ ]:

def loadEval(sess,conf,iterNum):
    outfilename = os.path.join(conf.cachedir,conf.evaloutname)
    ckptfilename = '%s-%d'%(outfilename,iterNum)
    print('Loading base from %s'%(ckptfilename))
    self.basesaver.restore(sess,ckptfilename)


# In[ ]:

def restoreEval(sess,conf,evalSaver,restore=True):
    outfilename = os.path.join(conf.cachedir,conf.evaloutname)
    latest_ckpt = tf.train.get_checkpoint_state(conf.cachedir,
                                        latest_filename = conf.evalckptname)
    if not latest_ckpt or not restore:
        startat = 0
        sess.run(tf.initialize_variables(PoseTools.getvars('poseEval')))
        print("Not loading eval variables. Initializing them")
        didRestore = False
    else:
        evalSaver.restore(sess,latest_ckpt.model_checkpoint_path)
        matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
        startat = int(matchObj.group(1))+1
        print("Loading eval variables from %s"%latest_ckpt.model_checkpoint_path)
        didRestore = True
        
    return didRestore,startat


# In[ ]:

def saveEval(sess,step,evalSaver,conf):
    outfilename = os.path.join(conf.cachedir,conf.evaloutname)
    evalSaver.save(sess,outfilename,global_step=step,
               latest_filename = conf.evalckptname)


# In[ ]:

def genLabels(rlocs,locs,conf):
    d2locs = np.sqrt(((rlocs-locs[...,np.newaxis])**2).sum(-2))
    ll = np.arange(1,conf.n_classes+1)
    labels = np.tile(ll[:,np.newaxis],[d2locs.shape[0],1,d2locs.shape[2]])
    labels[d2locs>conf.poseEvalNegDist] = -1.
    labels[d2locs<conf.poseEvalNegDist] = 1.
    labels = np.concatenate([labels[:,np.newaxis],1-labels[:,np.newaxis]],-1)


# In[ ]:

def genRandomNegSamples(bout,l7out,locs,conf,nsamples=10):
    sz = (np.array(l7out.shape[1:3])-1)*conf.rescale*conf.pool_scale
    bsize = conf.batch_size
    rlocs = np.zeros(locs.shape + (nsamples,))
    rlocs[:,:,0,:] = np.random.randint(sz[1],size=locs.shape[0:2]+(nsamples,))
    rlocs[:,:,1,:] = np.random.randint(sz[0],size=locs.shape[0:2]+(nsamples,))
    return rlocs


# In[ ]:

def genGaussianPosSamples(bout,l7out,locs,conf,nsamples=10,maxlen = 4):
    scale = conf.rescale*conf.pool_scale
    sigma = float(maxlen)*0.5*scale
    sz = (np.array(l7out.shape[1:3])-1)*scale
    bsize = conf.batch_size
    rlocs = np.round(np.random.normal(size=locs.shape+(15*nsamples,))*sigma)
    # remove rlocs that are far away.
    dlocs = np.all( np.sqrt( (rlocs**2).sum(2))< (maxlen*scale),1)
    clocs = np.zeros(locs.shape+(nsamples,))
    for ii in range(dlocs.shape[0]):
        ndx = np.where(dlocs[ii,:])[0][:nsamples]
        clocs[ii,:,:,:] = rlocs[ii,:,:,ndx].transpose([1,2,0])

    rlocs = locs[...,np.newaxis] + clocs
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0,:]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0,:] = xlocs
    ylocs = rlocs[:,:,1,:]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1,:] = ylocs
    return rlocs


# In[ ]:

def genGaussianNegSamples(bout,l7out,locs,conf,nsamples=10,minlen = 8):
    scale = conf.rescale*conf.pool_scale
    sigma = minlen*scale
    sz = (np.array(l7out.shape[1:3])-1)*scale
    bsize = conf.batch_size
    rlocs = np.round(np.random.normal(size=locs.shape+(5*nsamples,))*sigma)
    # remove rlocs that are small.
    dlocs = np.sqrt( (rlocs**2).sum(2)).sum(1)
    clocs = np.zeros(locs.shape+(nsamples,))
    for ii in range(dlocs.shape[0]):
        ndx = np.where(dlocs[ii,:]> (minlen*conf.n_classes*scale) )[0][:nsamples]
        clocs[ii,:,:,:] = rlocs[ii,:,:,ndx].transpose([1,2,0])

    rlocs = locs[...,np.newaxis] + clocs
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0,:]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0,:] = xlocs
    ylocs = rlocs[:,:,1,:]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1,:] = ylocs
    return rlocs


# In[ ]:

def genMovedNegSamples(bout,l7out,locs,conf,nsamples=10,minlen=8):
    # Add same x and y to locs
    
    minlen = float(minlen)*1.25
    maxlen = 2*minlen
    rlocs = np.zeros(locs.shape + (nsamples,))
    sz = (np.array(l7out.shape[1:3])-1)*conf.rescale*conf.pool_scale

    for curi in range(locs.shape[0]):
        rx = np.round(np.random.rand(nsamples)*(maxlen-minlen) + minlen)*            np.sign(np.random.rand(nsamples)-0.5)
        ry = np.round(np.random.rand(nsamples)*(maxlen-minlen) + minlen)*            np.sign(np.random.rand(nsamples)-0.5)

        rlocs[curi,:,0,:] = locs[curi,:,0,np.newaxis] + rx*conf.rescale*conf.pool_scale
        rlocs[curi,:,1,:] = locs[curi,:,1,np.newaxis] + ry*conf.rescale*conf.pool_scale
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0,:]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0,:] = xlocs
    ylocs = rlocs[:,:,1,:]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1,:] = ylocs
    return rlocs


# In[ ]:

def genOneMovedNegSamples(bout,l7out,locs,conf,nsamples=10,minlen=8):
    # Move one of the points.
    minlen = 1.5*float(minlen)
    maxlen = 2*minlen
    
    rlocs = np.tile(locs[...,np.newaxis],[1,1,1,nsamples])
    sz = (np.array(l7out.shape[1:3])-1)*conf.rescale*conf.pool_scale

    for curi in range(locs.shape[0]):
        for curs in range(nsamples):
            rx = np.round(np.random.rand()*(maxlen-minlen) + minlen)*                np.sign(np.random.rand()-0.5)
            ry = np.round(np.random.rand()*(maxlen-minlen) + minlen)*                np.sign(np.random.rand()-0.5)
            rand_point = np.random.randint(conf.n_classes)
            
            rlocs[curi,rand_point,0,curs] = locs[curi,rand_point,0] + rx*conf.rescale*conf.pool_scale
            rlocs[curi,rand_point,1,curs] = locs[curi,rand_point,1] + ry*conf.rescale*conf.pool_scale
    
    # sanitize the locs
    rlocs[rlocs<0] = 0
    xlocs = rlocs[:,:,0,:]
    xlocs[xlocs>=sz[1]] = sz[1]-1
    rlocs[:,:,0,:] = xlocs
    ylocs = rlocs[:,:,1,:]
    ylocs[ylocs>=sz[0]] = sz[0]-1
    rlocs[:,:,1,:] = ylocs
    return rlocs


# In[ ]:

def genNegSamples(bout,l7out,locs,conf,nsamples=10):
    minlen = 5
    rlocs = np.concatenate([
#                           genRandomNegSamples(bout,l7out,locs,conf,nsamples),
                          genGaussianNegSamples(bout,l7out,locs,conf,nsamples,minlen),
                          genMovedNegSamples(bout,l7out,locs,conf,nsamples,minlen), 
                          genOneMovedNegSamples(bout,l7out,locs,conf,nsamples,minlen)], 
                          axis=3)
#     rlabels = genLabels(rlocs,locs,conf)
    return rlocs#,rlabels


# In[ ]:

def genData(l7out,inlocs,conf):
    locs = np.round(inlocs/(conf.rescale*conf.pool_scale))
    dd = np.zeros(locs.shape[0:2]+l7out.shape[-1:]+locs.shape[-1:])
    for curi in range(locs.shape[0]):
        for pp in range(locs.shape[1]):
            for s in range(locs.shape[3]):
                dd[curi,pp,:,s] = l7out[curi,int(locs[curi,pp,1,s]),int(locs[curi,pp,0,s]),:]
    return dd        


# In[ ]:

def prepareOpt(baseNet,dbtype,feed_dict,sess,conf,phDict,distort):
    nsamples = 10
    npos = 2
    baseNet.updateFeedDict(dbtype,distort)
    locs = baseNet.locs
    l7 = baseNet.baseLayers['conv7']
    [bout,l7out] = sess.run([baseNet.basePred,l7],feed_dict=baseNet.feed_dict)
    neglocs = genNegSamples(bout,l7out,locs,conf,nsamples=nsamples)
#     replocs = np.tile(locs[...,np.newaxis],npos*nsamples)
    replocs = genGaussianPosSamples(bout,l7out,locs,conf,nsamples*npos,maxlen=4)
    alllocs = np.concatenate([neglocs,replocs],axis=-1)
    alllocs = alllocs.transpose([0,3,1,2])
    alllocs = alllocs.reshape((-1,)+alllocs.shape[2:])
    retlocs = copy.deepcopy(alllocs)
    alllocs_m = alllocs.mean(1)
    alllocs = alllocs-alllocs_m[:,np.newaxis,:]
    
#    poslabels = np.ones(replocs.shape[0,1,3])
#    alllabels = np.concatenate([neglabels,poslabels],axis=-2)
#    alllabels = alllabels.transpose([0,2,1,3])
#    alllabels = alllabels.reshape((-1,)+alllabels.shape[-1])

    negdd = genData(l7out,neglocs,conf)
    posdd = genData(l7out,replocs,conf)
    alldd = np.concatenate([negdd,posdd],axis=-1)
    alldd = alldd.transpose([0,3,1,2])
    alldd = np.reshape(alldd,[-1,alldd.shape[-2],alldd.shape[-1]])

#     y = alllabels
    y = np.zeros([l7out.shape[0],neglocs.shape[-1]+replocs.shape[-1],2])
    y[:,:-nsamples*npos,0] = 1. 
    y[:,-nsamples*npos:,1] = 1.
    y = np.reshape(y,[-1,y.shape[-1]])
        
#     excount = step*conf.batch_size
#     cur_lr = learning_rate * \
#             conf.gamma**math.floor(excount/conf.step_size)
#     feed_dict[phDict['learning_rate']] = cur_lr
    feed_dict[phDict['y']] = y
    feed_dict[phDict['lin']] = alldd
    feed_dict[phDict['locs']] = alllocs
    return retlocs


# In[ ]:

def train(conf,restore=True):
    
    phDict = createEvalPH(conf)
    feed_dict = createFeedDict(phDict)
    feed_dict[phDict['phase_train']] = True
    feed_dict[phDict['dropout']] = 0.5
    with tf.variable_scope('poseEval'):
        out,layer_dict = poseEvalNet(phDict['lin'],phDict['locs'],
                                     conf,phDict['phase_train'],
                                     phDict['dropout'])
        
    evalSaver = createEvalSaver(conf)
    y = phDict['y']
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(out,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    baseNet = PoseTools.createNetwork(conf,1)
    baseNet.openDBs()
    
    with baseNet.env.begin() as txn,baseNet.valenv.begin() as valtxn,tf.Session() as sess:

        baseNet.createCursors()
        baseNet.restoreBase(sess,True)
        didRestore,startat = restoreEval(sess,conf,evalSaver,restore)
        baseNet.initializeRemainingVars(sess)
        for step in range(startat,conf.eval_training_iters+1):
            prepareOpt(baseNet,baseNet.DBType.Train,feed_dict,sess,conf,
                       phDict,distort=True)
#             baseNet.feed_dict[baseNet.ph['keep_prob']] = 0.5
            feed_dict[phDict['phase_train']] = True
            feed_dict[phDict['dropout']] = 0.5
            sess.run(train_step, feed_dict=feed_dict)

            if step % 25 == 0:
                prepareOpt(baseNet,baseNet.DBType.Train,feed_dict,
                           sess,conf,phDict,distort=False)
#                 baseNet.feed_dict[baseNet.ph['keep_prob']] = 1
                feed_dict[phDict['phase_train']] = False
                feed_dict[phDict['dropout']] = 1
                train_cross_ent = sess.run(cross_entropy, feed_dict=feed_dict)
                test_cross_ent = 0
                test_acc = 0
                pos_acc = 0
                pred_acc_pos = 0
                pred_acc_pred = 0
                
                #generate blocks for different neg types
                nsamples = 10
                ntypes = 3
                npos = 2
                blk = np.arange(nsamples)
                inter = np.arange(0,conf.batch_size*nsamples*(ntypes+npos),nsamples*(ntypes+npos))
                n_inters = blk + inter[:,np.newaxis]
                n_inters = n_inters.flatten()
                nacc = np.zeros(ntypes)
                nclose = 0 
                in_locs = np.array([])
                nrep = 40
                for rep in range(nrep):
                    prepareOpt(baseNet,baseNet.DBType.Val,feed_dict,sess,conf,
                               phDict,distort=False)
                    test_cross_ent += sess.run(cross_entropy, feed_dict=feed_dict)
                    test_acc += sess.run(accuracy, feed_dict = feed_dict)
                    tout = sess.run(correct_prediction, feed_dict=feed_dict)
                    labels = feed_dict[phDict['y']]
                    pos_acc += float(np.count_nonzero(tout[labels[:,1]>0.5]))/nsamples
                    for nt in range(ntypes):
                        nacc[nt] += float(np.count_nonzero(tout[n_inters+nt*nsamples]))/nsamples
                    
                    tdd = feed_dict[phDict['lin']]
                    tlocs = feed_dict[phDict['locs']]
                    ty = feed_dict[phDict['y']]
                    # 
                    l7 = baseNet.baseLayers['conv7']
                    curpred = sess.run([baseNet.basePred,l7], feed_dict=baseNet.feed_dict)
                    baseLocs = PoseTools.getBasePredLocs(curpred[0],conf)

                    neglocs = baseLocs[:,:,:,np.newaxis]
                    locs = np.array(baseNet.locs)[...,np.newaxis]
                    d2locs = np.sqrt( np.sum((neglocs-locs)**2,axis=(1,2,3)))
                    alllocs = np.concatenate([neglocs,locs],axis=3)
                    alldd = genData(curpred[1],alllocs,conf)
                    alllocs = alllocs.transpose([0,3,1,2])
                    alllocs = alllocs.reshape((-1,)+alllocs.shape[2:])
                    alllocs_m = alllocs.mean(1)
                    alllocs = alllocs-alllocs_m[:,np.newaxis,:]

                    alldd = alldd.transpose([0,3,1,2])
                    alldd = np.reshape(alldd,[-1,alldd.shape[-2],alldd.shape[-1]])

                    y = np.zeros([curpred[0].shape[0],alllocs.shape[-1],2])
                    y[d2locs>=25,:-1,0] = 1. 
                    y[d2locs<25,:-1,1] = 1. 
                    y[:,-1,1] = 1.
                    y = np.reshape(y,[-1,y.shape[-1]])

                    feed_dict[phDict['y']] = y
                    feed_dict[phDict['lin']] = alldd
                    feed_dict[phDict['locs']] = alllocs

                    corrpred = sess.run(correct_prediction,feed_dict=feed_dict)
                    pred_acc_pos += np.count_nonzero(corrpred[1::2])
                    pred_acc_pred += np.count_nonzero(corrpred[0::2])
                    nclose += np.count_nonzero(d2locs<25)
                    er_locs = ~corrpred[0::2]
                    in_locs = np.append(in_locs,d2locs[er_locs])
                print "Iter:{:d}, train:{:.4f} test:{:.4f} acc:{:.2f} posacc:{:.2f}".format(step,
                                                 train_cross_ent,test_cross_ent/nrep,
                                                 test_acc/nrep,pos_acc/nrep/conf.batch_size/npos)
                print "Neg:{}".format(nacc/nrep/conf.batch_size)
                print "Pred Acc Pos:{},Pred Acc Pred:{},numclose:{}".format(float(pred_acc_pos)/nrep/conf.batch_size,
                                                                            float(pred_acc_pred)/nrep/conf.batch_size,
                                                                            float(nclose)/nrep/conf.batch_size)
                print 'Distance of incorrect predictions:{}'.format(in_locs)
                
            if step % 100 == 0:
                saveEval(sess,step,evalSaver,conf)

