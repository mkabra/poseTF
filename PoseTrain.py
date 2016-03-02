
# coding: utf-8

# In[ ]:

'''
Mayank Feb 3 2016
'''

import tensorflow as tf
import os,sys,shutil
import tempfile,copy,re
from enum import Enum

sys.path.append('/home/mayank/work/caffe/python')
sys.path.append('/home/mayank/work/pyutils')

import caffe,lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math,cv2,scipy,time,pickle
import numpy as np

import PoseTools,myutils,multiResData

import convNetBase as CNB


# In[ ]:

class PoseTrain:
    
    Nets = Enum('Nets','Base Joint Fine')
    TrainingType = Enum('TrainingType','Base Joint Fine All')
    DBType = Enum('DBType','Train Val')
    
    def __init__(self,conf,nettype):
        self.conf = conf
        self.nettype = nettype
        self.feed_dict = {}
    
    def openDBs(self):
        lmdbfilename =os.path.join(self.conf.cachedir,self.conf.trainfilename)
        vallmdbfilename =os.path.join(self.conf.cachedir,self.conf.valfilename)
        self.env = lmdb.open(lmdbfilename, readonly = True)
        self.valenv = lmdb.open(vallmdbfilename, readonly = True)

    def createCursors(self,txn,valtxn):
        self.train_cursor = txn.cursor(); 
        self.val_cursor = valtxn.cursor()
        
    def createPH(self):
        x0,x1,x2,y,keep_prob = CNB.createPlaceHolders(self.conf.imsz,
                               self.conf.rescale,self.conf.scale,self.conf.pool_scale,
                                self.conf.n_classes)
        locs_ph = tf.placeholder(tf.float32,[self.conf.batch_size,
                                             self.conf.n_classes,2])
        learning_rate_ph = tf.placeholder(tf.float32,shape=[])
        self.ph = {'x0':x0,'x1':x1,'x2':x2,
                     'y':y,'keep_prob':keep_prob,'locs':locs_ph,
                     'learning_rate':learning_rate_ph}
        
        
    def readImages(self,dbType):
        conf = self.conf
        curcursor = self.val_cursor if (dbType == self.DBType.Val)                     else self.train_cursor
        xs, locs = PoseTools.readLMDB(curcursor,
                         conf.batch_size,conf.imsz,multiResData)
        locs = multiResData.sanitizelocs(locs)
        self.xs = xs
        self.locs = locs
        

    def createFeedDict(self):
        self.feed_dict = {self.ph['x0']:[],
                          self.ph['x1']:[],
                          self.ph['x2']:[],
                          self.ph['y']:[],
                          self.ph['keep_prob']:1.,
                          self.ph['learning_rate']:1,
                          self.ph['locs']:[]}

    def updateFeedDict(self,dbType):
        conf = self.conf
        self.readImages(dbType)
        x0,x1,x2 = PoseTools.multiScaleImages(
            self.xs.transpose([0,2,3,1]),conf.rescale,conf.scale)

        labelims = PoseTools.createLabelImages(self.locs,
                                   self.conf.imsz,
                                   self.conf.pool_scale*self.conf.rescale,
                                   self.conf.label_blur_rad)
        self.feed_dict[self.ph['x0']] = x0
        self.feed_dict[self.ph['x1']] = x1
        self.feed_dict[self.ph['x2']] = x2
        self.feed_dict[self.ph['y']] = labelims
        self.feed_dict[self.ph['locs']] = self.locs
        
    
    def createBaseNetwork(self):
        pred,layers = CNB.net_multi_conv(self.ph['x0'],
                                     self.ph['x1'],
                                     self.ph['x2'],
                                     self.ph['keep_prob'],self.conf)
        self.basePred = pred
        self.baseLayers = layers

        
    def createFineNetwork(self):
        if self.curtrainingType == self.TrainingType.All:
            pred = tf.stop_gradient(self.basePred)
        else:
            pred = self.basePred

        # Construct fine model
        labelT  = PoseTools.createFineLabelTensor(self.conf)
        layers = self.baseLayers
        layer1_1 = tf.stop_gradient(layers['base_dict_0']['conv1'])
        layer1_2 = tf.stop_gradient(layers['base_dict_0']['conv2'])
        layer2_1 = tf.stop_gradient(layers['base_dict_1']['conv1'])
        layer2_2 = tf.stop_gradient(layers['base_dict_1']['conv2'])
        curfine1_1 = CNB.extractPatches(layer1_1,pred,self.conf,1,4)
        curfine1_2 = CNB.extractPatches(layer1_2,pred,self.conf,2,2)
        curfine2_1 = CNB.extractPatches(layer2_1,pred,self.conf,2,2)
        curfine2_2 = CNB.extractPatches(layer2_2,pred,self.conf,4,1)
        curfine1_1u = tf.unpack(tf.transpose(curfine1_1,[1,0,2,3,4]))
        curfine1_2u = tf.unpack(tf.transpose(curfine1_2,[1,0,2,3,4]))
        curfine2_1u = tf.unpack(tf.transpose(curfine2_1,[1,0,2,3,4]))
        curfine2_2u = tf.unpack(tf.transpose(curfine2_2,[1,0,2,3,4]))
        finepred = CNB.fineOut(curfine1_1u,curfine1_2u,curfine2_1u,curfine2_2u,self.conf)    
        limgs = PoseTools.createFineLabelImages(self.conf.ph['locs'],
                                                    pred,self.conf,labelT)
        
        self.finePred = finepred
        self.fine_labels = limgs

    def createBaseSaver(self):
        self.basesaver = tf.train.Saver(max_to_keep=self.conf.maxckpt)
        
    def loadBase(self):
        baseoutname = '%s_%d.ckpt'%(self.conf.outname,self.conf.base_training_iters)
        basemodelfile = os.path.join(self.conf.cachedir,baseoutname)

        self.basesaver.restore(self.sess,basemodelfile)

    def saveBase(self,sess,step):
        
        outfilename = os.path.join(self.conf.cachedir,self.conf.outname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.databasename)
        self.basesaver.save(sess,outfilename,global_step=step,
                   latest_filename = self.conf.ckptbasename)
        print('Saved state to %s-%d' %(outfilename,step))
#         if step%self.conf.back_steps==0:
#             shutil.copyfile('%s-%d'%(outfilename,step),'%s-%d.save'%(outfilename,step))
#             print('Backed up state to %s-%d.save' %(outfilename,step))
            
        with open(traindatafilename,'wb') as tdfile:
            pickle.dump(self.basetrainData,tdfile)

    def restoreBase(self,sess,restore):
        outfilename = os.path.join(self.conf.cachedir,self.conf.outname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.databasename)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                            latest_filename = self.conf.ckptbasename)
        if not latest_ckpt or not restore:
            self.basestartat = 0
            self.basetrainData = {'train_err':[],'val_err':[],'step_no':[]}
            init = tf.initialize_all_variables()
            sess.run(init)
            return False
        else:
            self.basesaver.restore(sess,latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
            self.basestartat = int(matchObj.group(1))+1
            with open(traindatafilename,'rb') as tdfile:
                self.basetrainData = pickle.load(tdfile)
            return True
            

    def createOptimizer(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=                           self.ph['learning_rate']).minimize(self.cost)
        self.read_time = 0.
        self.opt_time = 0.

    def doOpt(self,sess,step):
        excount = step*self.conf.batch_size
        cur_lr = self.conf.learning_rate *                 self.conf.gamma**math.floor(excount/self.conf.step_size)
        self.feed_dict[self.ph['learning_rate']] = cur_lr
        self.feed_dict[self.ph['keep_prob']] = self.conf.dropout
        r_start = time.clock()
        self.updateFeedDict(self.DBType.Train)
        r_end = time.clock()
        sess.run(self.opt, self.feed_dict)
        o_end = time.clock()
        
        self.read_time += r_end-r_start
        self.opt_time += o_end-r_end

    def computeLoss(self,sess,costfcns):
        self.feed_dict[self.ph['keep_prob']] = 1.
        loss = sess.run(costfcns,self.feed_dict)
        loss = [x/self.conf.batch_size for x in loss]
        return loss
        
    def updateBaseLoss(self,step,train_loss,val_loss):
        print "Iter " + str(step)
        print "  Training Loss= " + "{:.6f}".format(train_loss) +              ", Minibatch Loss= " + "{:.6f}".format(val_loss) 
        nstep = step-self.basestartat
        print "  Read Time:" + "{:.2f}, ".format(self.read_time/(nstep+1)) +               "Opt Time:" + "{:.2f}".format(self.opt_time/(nstep+1)) 
        self.basetrainData['train_err'].append(train_loss)        
        self.basetrainData['val_err'].append(val_loss)        
        self.basetrainData['step_no'].append(step)        

        
    def baseTrain(self,restore=True):
        self.createPH()
        self.createFeedDict()
        with tf.variable_scope('base'):
            self.createBaseNetwork()
        self.cost = tf.nn.l2_loss(self.basePred-self.ph['y'])
        self.openDBs()
        self.createOptimizer()
        self.createBaseSaver()
        with self.env.begin() as txn,self.valenv.begin() as valtxn, tf.Session() as sess:
            self.createCursors(txn,valtxn)
            self.restoreBase(sess,restore)
            for step in range(self.basestartat,self.conf.base_training_iters):
                self.doOpt(sess,step)
                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train)
                    train_loss = self.computeLoss(sess,[self.cost])[0]
                    numrep = int(self.conf.numTest/self.conf.batch_size)+1
                    val_loss = 0
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val)
                        val_loss += self.computeLoss(sess,[self.cost])[0]
                    val_loss = val_loss/numrep
                    self.updateBaseLoss(step,train_loss,val_loss)
                if step % self.conf.save_step == 0:
                    self.saveBase(sess,step)
            print("Optimization Finished!")
            self.saveBase(sess,step)
    
    def createNetwork(self):
        self.createPH()
        self.createFeedDict()
        with tf.variable_scope('base'):
            self.createBaseNetwork()
        
        

    def createMRFNetwork(self,passGradients=False):
        
        ncls = self.conf.n_classes
        ksz = self.conf.mrf_psz
        bpred = self.basePred if passGradients else tf.stop_gradient(self.basePred)
        mrf_weights = PoseTools.initMRFweights(conf)
        mrf_conv = 0
        for cls in range(ncls):
            with variable_scope('mrf_%d'%cls)
                weights = tf.get_variable("weights",
                              initializer=tf.constant(mrf_weights[:,:,cls:cls+1,:]))
                biases = tf.get_variable("biases", kernel_shape[-1],
                              initializer=tf.constant_initializer(0))

            sweights = tf.nn.softplus(weights)
            sbiases = tf.nn.softplus(biases)

            curconv = tf.nn.conv2d(bpred[:,:,:,cls], sweights,strides=[1, 1, 1, 1], padding='SAME')+sbiases
            mrf_conv += tf.log(curconv)
        
        self.mrfPred = tf.exp(mrf_conv)
        
    def mrfTrain(self,restore=True):
        self.createPH()
        self.createFeedDict()
        with tf.variable_scope('mrf'):
            self.createMRFNetwork()
        self.cost = tf.nn.l2_loss(self.mrfPred-self.ph['y'])
        self.openDBs()
        self.createOptimizer()
        self.createMRFSaver()
        with self.env.begin() as txn,self.valenv.begin() as valtxn, tf.Session() as sess:
            self.createCursors(txn,valtxn)
            self.restoreBase(sess,restore)
            for step in range(self.basestartat,self.conf.base_training_iters):
                self.doOpt(sess,step)
                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train)
                    train_loss = self.computeLoss(sess,[self.cost])[0]
                    numrep = int(self.conf.numTest/self.conf.batch_size)+1
                    val_loss = 0
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val)
                        val_loss += self.computeLoss(sess,[self.cost])[0]
                    val_loss = val_loss/numrep
                    self.updateMRFLoss(step,train_loss,val_loss)
                if step % self.conf.save_step == 0:
                    self.saveMRF(sess,step)
            print("Optimization Finished!")
            self.saveBase(sess,step)

        

