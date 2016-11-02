
# coding: utf-8

# In[ ]:

'''
Mayank Feb 3 2016
'''

import tensorflow as tf
import os,sys,shutil
import tempfile,copy,re
from enum import Enum
import localSetup

# import caffe,lmdb
# import caffe.proto.caffe_pb2
# from caffe.io import datum_to_array

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import math,cv2,scipy,time,pickle
import numpy as np

import PoseTools,myutils,multiResData

import convNetBase as CNB
from batch_norm import batch_norm


# In[ ]:

class PoseTrain(object):
    
    Nets = Enum('Nets','Base Joint Fine')
    TrainingType = Enum('TrainingType','Base MRF Fine All')
    DBType = Enum('DBType','Train Val')
   
    
    def __init__(self,conf):
        self.conf = conf
        self.feed_dict = {}
    
    # ---------------- DATABASE ---------------------
    
    def openDBs(self):
#         lmdbfilename =os.path.join(self.conf.cachedir,self.conf.trainfilename)
#         vallmdbfilename =os.path.join(self.conf.cachedir,self.conf.valfilename)
#         self.env = lmdb.open(lmdbfilename, readonly = True)
#         self.valenv = lmdb.open(vallmdbfilename, readonly = True)
        if self.trainType == 0:
            trainfilename =os.path.join(self.conf.cachedir,self.conf.trainfilename) + '.tfrecords'
            valfilename =os.path.join(self.conf.cachedir,self.conf.valfilename) + '.tfrecords'
            self.train_queue = tf.train.string_input_producer([trainfilename])
            self.val_queue = tf.train.string_input_producer([valfilename])
        else:
            trainfilename =os.path.join(self.conf.cachedir,self.conf.fulltrainfilename) + '.tfrecords'
            valfilename =os.path.join(self.conf.cachedir,self.conf.fulltrainfilename) + '.tfrecords'
            self.train_queue = tf.train.string_input_producer([trainfilename])
            self.val_queue = tf.train.string_input_producer([valfilename])
            

    def openHoldoutDBs(self):
        lmdbfilename =os.path.join(self.conf.cachedir,self.conf.holdouttrain)
        vallmdbfilename =os.path.join(self.conf.cachedir,self.conf.holdouttest)
        self.env = lmdb.open(lmdbfilename, readonly = True)
        self.valenv = lmdb.open(vallmdbfilename, readonly = True)

    def createCursors(self, sess, txn=None, valtxn=None):
#         if txn is None:
#             txn = self.env.begin()
#             valtxn = self.valenv.begin()
#         self.train_cursor = txn.cursor(); 
#         self.val_cursor = valtxn.cursor()
        train_ims,train_locs = multiResData.read_and_decode(self.train_queue,self.conf)
        val_ims,val_locs = multiResData.read_and_decode(self.val_queue,self.conf)
        self.train_data = [train_ims,train_locs]
        self.val_data = [val_ims,val_locs]
        coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
    def closeCursors(self):
        self.env.close()
        self.valenv.close()
        
    def readImages(self,dbType,distort,sess):
        conf = self.conf
#         curcursor = self.val_cursor if (dbType == self.DBType.Val) \
#                     else self.train_cursor
#         xs, locs = PoseTools.readLMDB(curcursor,
#                          conf.batch_size,conf.imsz,multiResData)
        cur_data = self.val_data if (dbType==self.DBType.Val)                 else self.train_data
        xs = []; locs = []
        for ndx in range(conf.batch_size):
            [curxs,curlocs] = sess.run(cur_data)
            if np.ndim(curxs)<3:
                xs.append(curxs[np.newaxis,:,:])
            else:
                xs.append(curxs)
            locs.append(curlocs)
        xs = np.array(xs)    
        locs = np.array(locs)
        locs = multiResData.sanitizelocs(locs)
        if distort:
            if conf.horzFlip:
                xs,locs = PoseTools.randomlyFlipLR(xs,locs)
            if conf.vertFlip:
                xs,locs = PoseTools.randomlyFlipUD(xs,locs)
            xs,locs = PoseTools.randomlyRotate(xs,locs,conf)
            xs = PoseTools.randomlyAdjust(xs,conf)
        
        self.xs = xs
        self.locs = locs
        
    def createPH(self):
        x0,x1,x2,y,keep_prob = CNB.createPlaceHolders(self.conf.imsz,
                               self.conf.rescale,self.conf.scale,self.conf.pool_scale,
                                self.conf.n_classes)
        locs_ph = tf.placeholder(tf.float32,[self.conf.batch_size,
                                             self.conf.n_classes,2])
        learning_rate_ph = tf.placeholder(tf.float32,shape=[],name='learning_r')
        phase_train_base = tf.placeholder(tf.bool, name='phase_train_base')                 
        phase_train_fine = tf.placeholder(tf.bool, name='phase_train_fine')                 
        self.ph = {'x0':x0,'x1':x1,'x2':x2,
                     'y':y,'keep_prob':keep_prob,'locs':locs_ph,
                     'phase_train_base':phase_train_base,
                     'phase_train_fine':phase_train_fine,
                     'learning_rate':learning_rate_ph}

    def createFeedDict(self):
        self.feed_dict = {self.ph['x0']:[],
                          self.ph['x1']:[],
                          self.ph['x2']:[],
                          self.ph['y']:[],
                          self.ph['keep_prob']:1.,
                          self.ph['learning_rate']:1,
                          self.ph['phase_train_base']:False,
                          self.ph['phase_train_fine']:False,
                          self.ph['locs']:[]}

    def updateFeedDict(self,dbType,distort,sess):
        conf = self.conf
        self.readImages(dbType,distort,sess)
        x0,x1,x2 = PoseTools.multiScaleImages(self.xs.transpose([0,2,3,1]),
                                              conf.rescale, conf.scale,
                                              conf.l1_cropsz,conf)

        labelims = PoseTools.createLabelImages(self.locs,
                                   self.conf.imsz,
                                   self.conf.pool_scale*self.conf.rescale,
                                   self.conf.label_blur_rad)
        self.feed_dict[self.ph['x0']] = x0
        self.feed_dict[self.ph['x1']] = x1
        self.feed_dict[self.ph['x2']] = x2
        self.feed_dict[self.ph['y']] = labelims
        self.feed_dict[self.ph['locs']] = self.locs
        

    # ---------------- NETWORK ---------------------
    
    def createBaseNetwork(self,doBatch):
        pred,layers = CNB.net_multi_conv(self.ph['x0'],self.ph['x1'],
                                         self.ph['x2'],self.ph['keep_prob'],
                                         self.conf,doBatch,
                                         self.ph['phase_train_base']
                                        )
        self.basePred = pred
        self.baseLayers = layers

    def createACNetwork(self,doBatch,jointTraining=False):
        
        n_classes = self.conf.n_classes
        with tf.variable_scope('AC_'):
            self.createMRFNetwork(doBatch,jointTraining)
        
        mrfpred = self.mrfPred if jointTraining else tf.stop_gradient(self.mrfPred)
        
        layer7_init = self.baseLayers['conv7']
        layer7 = layer7_init if jointTraining else tf.stop_gradient(layer7_init)
        
        with tf.variable_scope('AC_'):
            ac_weights = tf.get_variable("weights", 
                [1,1,self.conf.nfcfilt,self.conf.n_classes],
                initializer=tf.random_normal_initializer(stddev=0.01))
            ac_biases = tf.get_variable("biases", self.conf.n_classes,
                initializer=tf.constant_initializer(0))
            ac_out = tf.nn.conv2d(layer7, ac_weights,
                strides=[1, 1, 1, 1], padding='SAME') + ac_biases
        self.acPred = ac_out + mrfpred
        
    def createMRFNetwork(self,doBatch,jointTraining=False):
        
        n_classes = self.conf.n_classes
        bpred = self.basePred if jointTraining else tf.stop_gradient(self.basePred)
        mrf_weights = PoseTools.initMRFweights(self.conf).astype('float32')
        mrf_weights = mrf_weights/n_classes
        
        baseShape = tf.Tensor.get_shape(bpred).as_list()[1:3]
        mrf_sz = mrf_weights.shape[0:2]

        bpred = tf.nn.relu((bpred+1)/2)
        sliceEnd = [0,0]
        pad = False
        if mrf_sz[0] > baseShape[0]:
            dd1 = int(math.ceil(float(mrf_sz[0]-baseShape[0])/2))
            sliceEnd[0] = mrf_sz[0]-dd1
            pad = True
        else:
            dd1 = 0
            sliceEnd[0] = baseShape[0]
            
        if mrf_sz[1] > baseShape[1]:
            dd2 = int(math.ceil(float(mrf_sz[1]-baseShape[1])/2))
            sliceEnd[1] = mrf_sz[1]-dd2
            pad = True
        else:
            dd2 = 0
            sliceEnd[1] = baseShape[1]
            
        if pad:    
            print('Padding base prediction by %d,%d. Filter shape:%d, Base shape:%d'%(dd1,dd2,mrf_sz[0],mr,baseShape))
            bpred = tf.pad(bpred,[[0,0],[dd1,dd1],[dd2,dd2],[0,0]])

        ksz = math.sqrt(mrf_weights.shape[0]*mrf_weights.shape[1])
        with tf.variable_scope('mrf'):
            conv_std = (1./n_classes)/ksz
            weights = tf.get_variable("weights", initializer=tf.constant(mrf_weights))
            biases = tf.get_variable("biases", n_classes,
                                     initializer=tf.constant_initializer(0))
            conv = tf.nn.conv2d(bpred, weights,
                               strides=[1, 1, 1, 1], padding='SAME')
            mrfout = conv+biases
        self.mrfPred = mrfout[:,dd1:sliceEnd[0],dd2:sliceEnd[1],:]
        
#   BELOW is from Chris bregler's paper where they try to do MRF style inference.
#   This didn't work so well and the performance was always worse than base.
#         mrf_conv = 0
#         conv_out = []
#         all_wts = []
#         for cls in range(n_classes):
#             with tf.variable_scope('mrf_%d'%cls):
#                 curwt = 10*mrf_weights[:,:,cls:cls+1,:]-3
#                 #the scaling is so that zero values are close to zero after softplus
#                 weights = tf.get_variable("weights",dtype = tf.float32,
#                               initializer=tf.constant(curwt))
#                 biases = tf.get_variable("biases", mrf_weights.shape[-1],dtype = tf.float32,
#                               initializer=tf.constant_initializer(-1))

#             sweights = tf.nn.softplus(weights)
#             sbiases = tf.nn.softplus(biases)
            
#             if doBatch:
#                 curBasePred = batch_norm(bpred[:,:,:,cls:cls+1],trainPhase)
#                 curBasePred = tf.maximum(curBasePred,0.0001)
#             else:
#                 curBasePred = tf.maximum(bpred[:,:,:,cls:cls+1],0.0001)
                
#             curconv = tf.nn.conv2d(curBasePred,sweights,strides=[1, 1, 1, 1], 
#                                    padding='SAME')+sbiases
#             conv_out.append(tf.log(curconv))
#             mrf_conv += tf.log(curconv)
#             all_wts.append(sweights)
#         mrfout = tf.exp(mrf_conv)
#         self.mrfPred = mrfout[:,dd:sliceEnd,dd:sliceEnd,:]
#    Return the value below when we used MRF kind of stuff
#         return conv_out,all_wts
        
    def createFineNetwork(self,doBatch,jointTraining=False):
        if self.conf.useMRF:
            if not jointTraining:
                pred = tf.stop_gradient(self.mrfPred)
            else:
                pred = self.mrfPred
        else:
            if not jointTraining:
                pred = tf.stop_gradient(self.basePred)
            else:
                pred = self.basePred

        # Construct fine model
        labelT  = PoseTools.createFineLabelTensor(self.conf)
        layers = self.baseLayers
        
        if not jointTraining:
            layer1_1 = tf.stop_gradient(layers['base_dict_0']['conv1'])
            layer1_2 = tf.stop_gradient(layers['base_dict_0']['conv2'])
            layer2_1 = tf.stop_gradient(layers['base_dict_1']['conv1'])
            layer2_2 = tf.stop_gradient(layers['base_dict_1']['conv2'])
        else:
            layer1_1 = layers['base_dict_0']['conv1']
            layer1_2 = layers['base_dict_0']['conv2']
            layer2_1 = layers['base_dict_1']['conv1']
            layer2_2 = layers['base_dict_1']['conv2']
            
        curfine1_1 = CNB.extractPatches(layer1_1,pred,self.conf,1,4)
        curfine1_2 = CNB.extractPatches(layer1_2,pred,self.conf,2,2)
        curfine2_1 = CNB.extractPatches(layer2_1,pred,self.conf,2,2)
        curfine2_2 = CNB.extractPatches(layer2_2,pred,self.conf,4,1)
        curfine1_1u = tf.unpack(tf.transpose(curfine1_1,[1,0,2,3,4]))
        curfine1_2u = tf.unpack(tf.transpose(curfine1_2,[1,0,2,3,4]))
        curfine2_1u = tf.unpack(tf.transpose(curfine2_1,[1,0,2,3,4]))
        curfine2_2u = tf.unpack(tf.transpose(curfine2_2,[1,0,2,3,4]))
        finepred = CNB.fineOut(curfine1_1u,curfine1_2u,curfine2_1u,curfine2_2u,
                               curfine7u,self.conf,doBatch,
                               self.ph['phase_train_fine'])    
        limgs = PoseTools.createFineLabelImages(self.ph['locs'],
                                                pred,self.conf,labelT)
        self.finePred = finepred
        self.fine_labels = limgs

    # ---------------- SAVING/RESTORING ---------------------

    def createBaseSaver(self):
        self.basesaver = tf.train.Saver(var_list = PoseTools.getvars('base'),
                                        max_to_keep=self.conf.maxckpt)
        
    def createACSaver(self):
        self.acsaver = tf.train.Saver(var_list = PoseTools.getvars('AC_'),
                                       max_to_keep=self.conf.maxckpt)
        
    def createMRFSaver(self):
        self.mrfsaver = tf.train.Saver(var_list = PoseTools.getvars('mrf'),
                                       max_to_keep=self.conf.maxckpt)
        
    def createFineSaver(self):
        self.finesaver = tf.train.Saver(var_list = PoseTools.getvars('fine'),
                                        max_to_keep=self.conf.maxckpt)
        
    def createJointSaver(self):
        vlist = PoseTools.getvars('fine') +                 PoseTools.getvars('mrf') +                 PoseTools.getvars('base') 
        self.jointsaver = tf.train.Saver(var_list = vlist,
                                        max_to_keep=self.conf.maxckpt)
        
    def loadBase(self,sess,iterNum):
        outfilename = os.path.join(self.conf.cachedir,self.conf.baseoutname)
        ckptfilename = '%s-%d'%(outfilename,iterNum)
        print('Loading base from %s'%(ckptfilename))
        self.basesaver.restore(sess,ckptfilename)
            
    def restoreBase(self,sess,restore):
        outfilename = os.path.join(self.conf.cachedir,self.conf.baseoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.basedataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                            latest_filename = self.conf.baseckptname)
        if not latest_ckpt or not restore:
            self.basestartat = 0
            self.basetrainData = {'train_err':[], 'val_err':[], 'step_no':[],
                                  'train_dist':[], 'val_dist':[] }
            sess.run(tf.initialize_variables(PoseTools.getvars('base')))
            print("Not loading base variables. Initializing them")
            return False
        else:
            self.basesaver.restore(sess,latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
            self.basestartat = int(matchObj.group(1))+1
            with open(traindatafilename,'rb') as tdfile:
                inData = pickle.load(tdfile)
                if not isinstance(inData,dict):
                    self.basetrainData, loadconf = inData
                    print('Parameters that dont match for base:')
                    PoseTools.compareConf(self.conf, loadconf)
                else:
                    print("No config was stored for base. Not comparing conf")
                    self.basetrainData = inData
            print("Loading base variables from %s"%latest_ckpt.model_checkpoint_path)
            return True
            
    def restoreAC(self,sess,restore):
        outfilename = os.path.join(self.conf.cachedir,self.conf.acoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.acdataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                            latest_filename = self.conf.acckptname)
        if not latest_ckpt or not restore:
            self.acstartat = 0
            self.actrainData = {'train_err':[],'val_err':[],'step_no':[],
                                'train_base_err':[],'val_base_err':[],
                                 'train_dist':[],'val_dist':[],
                                'train_base_dist':[],'val_base_dist':[]}
            sess.run(tf.initialize_variables(PoseTools.getvars('AC_')))
            print("Not loading AC variables. Initializing them")
            return False
        else:
            self.acsaver.restore(sess,latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
            self.acstartat = int(matchObj.group(1))+1
            with open(traindatafilename,'rb') as tdfile:
                inData = pickle.load(tdfile)
                if not isinstance(inData,dict):
                    self.actrainData, loadconf = inData
                    print('Parameters that dont match for AC:')
                    PoseTools.compareConf(self.conf, loadconf)
                else:
                    print("No config was stored for AC. Not comparing conf")
                    self.actrainData = inData
            print("Loading AC variables from %s"%latest_ckpt.model_checkpoint_path)
            return True
            
    def restoreMRF(self,sess,restore):
        outfilename = os.path.join(self.conf.cachedir,self.conf.mrfoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.mrfdataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                            latest_filename = self.conf.mrfckptname)
        if not latest_ckpt or not restore:
            self.mrfstartat = 0
            self.mrftrainData = {'train_err':[],'val_err':[],'step_no':[],
                                'train_base_err':[],'val_base_err':[],
                                 'train_dist':[],'val_dist':[],
                                'train_base_dist':[],'val_base_dist':[]}
            sess.run(tf.initialize_variables(PoseTools.getvars('mrf')))
            print("Not loading mrf variables. Initializing them")
            return False
        else:
            self.mrfsaver.restore(sess,latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
            self.mrfstartat = int(matchObj.group(1))+1
            with open(traindatafilename,'rb') as tdfile:
                inData = pickle.load(tdfile)
                if not isinstance(inData,dict):
                    self.mrftrainData, loadconf = inData
                    print('Parameters that dont match for mrf:')
                    PoseTools.compareConf(self.conf, loadconf)
                else:
                    print("No config was stored for mrf. Not comparing conf")
                    self.mrftrainData = inData
            print("Loading mrf variables from %s"%latest_ckpt.model_checkpoint_path)
            return True
            
    def restoreFine(self,sess,restore):
        outfilename = os.path.join(self.conf.cachedir,self.conf.fineoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.finedataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                            latest_filename = self.conf.fineckptname)
        if not latest_ckpt or not restore:
            self.finestartat = 0
            self.finetrainData = {'train_err':[],'val_err':[],'step_no':[],
                                'train_mrf_err':[],'val_mrf_err':[],
                                'train_base_err':[],'val_base_err':[],
                                'train_dist':[],'val_dist':[],
                                'train_mrf_dist':[],'val_mrf_dist':[],
                                'train_base_dist':[],'val_base_dist':[]}
            sess.run(tf.initialize_variables(PoseTools.getvars('fine')))
            print("Not loading fine variables. Initializing them")
            return False
        else:
            self.finesaver.restore(sess,latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
            self.finestartat = int(matchObj.group(1))+1
            with open(traindatafilename,'rb') as tdfile:
                inData = pickle.load(tdfile)
                if not isinstance(inData,dict):
                    self.finetrainData, loadconf = inData
                    print('Parameters that dont match for fine:')
                    PoseTools.compareConf(self.conf, loadconf)
                else:
                    print("No conf was stored for fine. Not comparing conf")
                    self.finetrainData = inData
            print("Loading fine variables from %s"%latest_ckpt.model_checkpoint_path)
            return True

    def restoreJoint(self,sess,restore):
        outfilename = os.path.join(self.conf.cachedir,self.conf.jointoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.jointdataname)
        latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                latest_filename = self.conf.jointckptname)
        if not latest_ckpt or not restore:
            self.finestartat = 0
            self.jointtrainData = {'train_err':[],'val_err':[],'step_no':[],
                                'train_fine_err':[],'val_fine_err':[],
                                'train_mrf_err':[],'val_mrf_err':[],
                                'train_base_err':[],'val_base_err':[],
                                'train_dist':[],'val_dist':[],
                                'train_fine_dist':[],'val_fine_dist':[],
                                'train_mrf_dist':[],'val_mrf_dist':[],
                                'train_base_dist':[],'val_base_dist':[]}
            print("Not loading joint variables. Initializing them")
            return False
        else:
            self.jointsaver.restore(sess,latest_ckpt.model_checkpoint_path)
            print("Loading joint variables from %s"%latest_ckpt.model_checkpoint_path)
            matchObj = re.match(outfilename + '-(\d*)',latest_ckpt.model_checkpoint_path)
            self.jointstartat = int(matchObj.group(1))+1
            with open(traindatafilename,'rb') as tdfile:
                inData = pickle.load(tdfile)
                if not isinstance(inData,dict):
                    self.jointtrainData, loadconf = inData
                    print('Parameters that dont match for joint:')
                    PoseTools.compareConf(self.conf, loadconf)
                else:
                    print("No conf was stored for joint. Not comparing conf")
                    self.jointtrainData = inData
            return True

        
    def saveBase(self,sess,step):
        outfilename = os.path.join(self.conf.cachedir,self.conf.baseoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.basedataname)
        self.basesaver.save(sess,outfilename,global_step=step,
                   latest_filename = self.conf.baseckptname)
        print('Saved state to %s-%d' %(outfilename,step))
        with open(traindatafilename,'wb') as tdfile:
            pickle.dump([self.basetrainData,self.conf],tdfile)
            
    def saveAC(self,sess,step):
        outfilename = os.path.join(self.conf.cachedir,self.conf.acoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.acdataname)
        self.acsaver.save(sess,outfilename,global_step=step,
                   latest_filename = self.conf.acckptname)
        print('Saved state to %s-%d' %(outfilename,step))
        with open(traindatafilename,'wb') as tdfile:
            pickle.dump([self.actrainData,self.conf],tdfile)

    def saveMRF(self,sess,step):
        outfilename = os.path.join(self.conf.cachedir,self.conf.mrfoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.mrfdataname)
        self.mrfsaver.save(sess,outfilename,global_step=step,
                   latest_filename = self.conf.mrfckptname)
        print('Saved state to %s-%d' %(outfilename,step))
        with open(traindatafilename,'wb') as tdfile:
            pickle.dump([self.mrftrainData,self.conf],tdfile)

    def saveFine(self,sess,step):
        outfilename = os.path.join(self.conf.cachedir,self.conf.fineoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.finedataname)
        self.finesaver.save(sess,outfilename,global_step=step,
                   latest_filename = self.conf.fineckptname)
        print('Saved state to %s-%d' %(outfilename,step))
        with open(traindatafilename,'wb') as tdfile:
            pickle.dump([self.finetrainData,self.conf],tdfile)

    def saveJoint(self,sess,step):
        outfilename = os.path.join(self.conf.cachedir,self.conf.jointoutname)
        traindatafilename = os.path.join(self.conf.cachedir,self.conf.jointdataname)
        self.finesaver.save(sess,outfilename,global_step=step,
                   latest_filename = self.conf.jointckptname)
        print('Saved state to %s-%d' %(outfilename,step))
        with open(traindatafilename,'wb') as tdfile:
            pickle.dump([self.jointtrainData,self.conf],tdfile)

    def initializeRemainingVars(self,sess):
        varlist = tf.all_variables()
        for var in varlist:
            try:
                sess.run(tf.assert_variables_initialized([var]))
            except tf.errors.FailedPreconditionError:
                sess.run(tf.initialize_variables([var]))
                print('Initializing variable:%s'%var.name)
                
                
    # ---------------- OPTIMIZATION/LOSS ---------------------

    
    
    def createOptimizer(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=                           self.ph['learning_rate']).minimize(self.cost)
        self.read_time = 0.
        self.opt_time = 0.

    def doOpt(self,sess,step,learning_rate):
        excount = step*self.conf.batch_size
        cur_lr = learning_rate *                 self.conf.gamma**math.floor(excount/self.conf.step_size)
        self.feed_dict[self.ph['learning_rate']] = cur_lr
        self.feed_dict[self.ph['keep_prob']] = self.conf.dropout
        r_start = time.clock()
        self.updateFeedDict(self.DBType.Train,distort=True,sess=sess)
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
    
    def computePredDist(self,sess,predfcn):
        self.feed_dict[self.ph['keep_prob']] = 1.
        pred = sess.run(predfcn,self.feed_dict)
        bee = PoseTools.getBaseError(self.locs,pred,self.conf)
        dee = np.sqrt(np.sum(np.square(bee),2))
        return dee
        
    def computeFinePredDist(self,sess,predfcn):
        self.feed_dict[self.ph['keep_prob']] = 1.
        pred = sess.run(predfcn,self.feed_dict)
        base_ee,fine_ee = PoseTools.getFineError(self.locs,pred[0],pred[1],self.conf)
        base_dist = np.sqrt(np.sum(np.square(base_ee),2))
        fine_dist = np.sqrt(np.sum(np.square(fine_ee),2))
        return fine_dist

    def updateBaseLoss(self,step,train_loss,val_loss,trainDist,valDist):
        print "Iter " + str(step) +              ", Train = " + "{:.3f},{:.1f}".format(train_loss[0],trainDist[0]) +              ", Val = " + "{:.3f},{:.1f}".format(val_loss[0],valDist[0])
#         nstep = step-self.basestartat
#         print "  Read Time:" + "{:.2f}, ".format(self.read_time/(nstep+1)) + \
#               "Opt Time:" + "{:.2f}".format(self.opt_time/(nstep+1)) 
        self.basetrainData['train_err'].append(train_loss[0])      
        self.basetrainData['val_err'].append(val_loss[0])        
        self.basetrainData['step_no'].append(step)        
        self.basetrainData['train_dist'].append(trainDist[0])        
        self.basetrainData['val_dist'].append(valDist[0])        

    def updateACLoss(self,step,train_loss,val_loss,trainDist,valDist):
        print "Iter " + str(step) +              ", Train = " + "{:.3f},{:.1f}".format(train_loss[0],trainDist[0]) +              ", Val = " + "{:.3f},{:.1f}".format(val_loss[0],valDist[0]) +              " ({:.1f},{:.1f}),({:.1f},{:.1f})".format(train_loss[1],val_loss[1],
                                                      trainDist[1],valDist[1])
        self.actrainData['train_err'].append(train_loss[0])        
        self.actrainData['val_err'].append(val_loss[0])        
        self.actrainData['train_base_err'].append(train_loss[1])        
        self.actrainData['val_base_err'].append(val_loss[1])        
        self.actrainData['train_dist'].append(trainDist[0])        
        self.actrainData['val_dist'].append(valDist[0])        
        self.actrainData['train_base_dist'].append(trainDist[1])        
        self.actrainData['val_base_dist'].append(valDist[1])        
        self.actrainData['step_no'].append(step)        

    def updateMRFLoss(self,step,train_loss,val_loss,trainDist,valDist):
        print "Iter " + str(step) +              ", Train = " + "{:.3f},{:.1f}".format(train_loss[0],trainDist[0]) +              ", Val = " + "{:.3f},{:.1f}".format(val_loss[0],valDist[0]) +              " ({:.1f},{:.1f}),({:.1f},{:.1f})".format(train_loss[1],val_loss[1],
                                                      trainDist[1],valDist[1])
        self.mrftrainData['train_err'].append(train_loss[0])        
        self.mrftrainData['val_err'].append(val_loss[0])        
        self.mrftrainData['train_base_err'].append(train_loss[1])        
        self.mrftrainData['val_base_err'].append(val_loss[1])        
        self.mrftrainData['train_dist'].append(trainDist[0])        
        self.mrftrainData['val_dist'].append(valDist[0])        
        self.mrftrainData['train_base_dist'].append(trainDist[1])        
        self.mrftrainData['val_base_dist'].append(valDist[1])        
        self.mrftrainData['step_no'].append(step)        

    def updateFineLoss(self,step,train_loss,val_loss,trainDist,valDist):
        print "Iter " + str(step) +              ", Train = " + "{:.3f},{:.1f}".format(train_loss[0],trainDist[0]) +              ", Val = " + "{:.3f},{:.1f}".format(val_loss[0],valDist[0]) +              " (MRF:{:.1f},{:.1f},{:.1f},{:.1f})".format(train_loss[1],val_loss[1],trainDist[1],valDist[1]) +              " (Base:{:.1f},{:.1f},{:.1f},{:.1f})".format(train_loss[2],val_loss[2],trainDist[2],valDist[2])
        self.finetrainData['train_err'].append(train_loss[0])        
        self.finetrainData['val_err'].append(val_loss[0])        
        self.finetrainData['train_mrf_err'].append(train_loss[1])        
        self.finetrainData['val_mrf_err'].append(val_loss[1])        
        self.finetrainData['train_base_err'].append(train_loss[2])        
        self.finetrainData['val_base_err'].append(val_loss[2])        
        self.finetrainData['step_no'].append(step)        
        self.finetrainData['train_dist'].append(trainDist[0])        
        self.finetrainData['val_dist'].append(valDist[0])        
        self.finetrainData['train_mrf_dist'].append(trainDist[1])        
        self.finetrainData['val_mrf_dist'].append(valDist[1])        
        self.finetrainData['train_base_dist'].append(trainDist[2])        
        self.finetrainData['val_base_dist'].append(valDist[2])        

    def updateJointLoss(self,step,train_loss,val_loss):
        print "Iter " + str(step) +              ", Train = " + "{:.3f}".format(train_loss[0]) +              ", Val = " + "{:.3f}".format(val_loss[0]) +              " (Fine:{:.1f},{:.1f})".format(train_loss[1],val_loss[1]) +              " (MRF:{:.1f},{:.1f})".format(train_loss[2],val_loss[2]) +              " (Base:{:.1f},{:.1f})".format(train_loss[3],val_loss[3])
        self.jointtrainData['train_err'].append(train_loss)        
        self.jointftrainData['val_err'].append(val_loss[0])        
        self.jointtrainData['train_fine_err'].append(train_loss[1])        
        self.jointtrainData['val_fine_err'].append(val_loss[1])        
        self.jointtrainData['train_mrf_err'].append(train_loss[2])        
        self.jointtrainData['val_mrf_err'].append(val_loss[2])        
        self.jointtrainData['train_base_err'].append(train_loss[3])        
        self.jointtrainData['val_base_err'].append(val_loss[3])        
        self.jointtrainData['step_no'].append(step)        

        

    # ---------------- TRAINING ---------------------

    
    
    def baseTrain(self, restore=True, trainPhase=True,trainType=0):
        self.createPH()
        self.createFeedDict()
        self.feed_dict[self.ph['phase_train_base']] = trainPhase
        self.feed_dict[self.ph['keep_prob']] = 0.5
        self.trainType = trainType
        doBatchNorm = self.conf.doBatchNorm
        
        with tf.variable_scope('base'):
            self.createBaseNetwork(doBatchNorm)
        self.cost = tf.nn.l2_loss(self.basePred-self.ph['y'])
        self.openDBs()
        self.createOptimizer()
        self.createBaseSaver()

#         with self.env.begin() as txn,\
#                  self.valenv.begin() as valtxn,\
        with tf.Session() as sess:
                    
            self.restoreBase(sess,restore)
            self.initializeRemainingVars(sess)
            self.createCursors(sess)
            
            for step in range(self.basestartat,self.conf.base_training_iters+1):
                self.feed_dict[self.ph['keep_prob']] = 0.5
                self.doOpt(sess,step,self.conf.base_learning_rate)
                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train,sess=sess,distort=True)
                    self.feed_dict[self.ph['keep_prob']] = 1.
                    train_loss = self.computeLoss(sess,[self.cost])
                    tt1 = self.computePredDist(sess,self.basePred)
                    trainDist = [tt1.mean()]
                    numrep = int(self.conf.numTest/self.conf.batch_size)+1
                    val_loss = np.zeros([2,])
                    valDist = [0.]
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val,distort=False,sess=sess)
                        val_loss += np.array(self.computeLoss(sess,[self.cost]))
                        tt1 = self.computePredDist(sess,self.basePred)
                        valDist = [valDist[0]+tt1.mean()]
                    val_loss = val_loss/numrep
                    valDist = [valDist[0]/numrep]
                    self.updateBaseLoss(step,train_loss,val_loss,trainDist,valDist)
                if step % self.conf.save_step == 0:
                    self.saveBase(sess,step)
            print("Optimization Finished!")
            self.saveBase(sess,step)
    
    
    def mrfTrain(self,restore=True,trainType=0):
        self.createPH()
        self.createFeedDict()
        doBatchNorm = self.conf.doBatchNorm
        self.feed_dict[self.ph['keep_prob']] = 0.5
        self.feed_dict[self.ph['phase_train_base']] = False
        self.trainType = trainType
        
        with tf.variable_scope('base'):
            self.createBaseNetwork(doBatchNorm)

        with tf.variable_scope('mrf'):
            self.createMRFNetwork(doBatchNorm)

        self.createBaseSaver()
        self.createMRFSaver()

        mod_labels = tf.maximum((self.ph['y']+1.)/2,0.01)
# the labels shouldn't be zero because the prediction is an output of
# exp. And it seems a lot of effort is wasted to make the prediction goto
# zero rather than match the location.
        
#         self.cost = tf.nn.l2_loss(self.mrfPred-mod_labels)
        self.cost = tf.nn.l2_loss(2*(self.mrfPred-0.5)-self.ph['y'])
        basecost  = tf.nn.l2_loss(self.basePred-self.ph['y'])
        
        if self.conf.useHoldout:
            self.openHoldoutDBs()
        else:
            self.openDBs()

        self.createOptimizer()

#         self.env.begin() as txn,self.valenv.begin() as valtxn,
        with tf.Session() as sess:
            
            self.loadBase(sess,self.conf.baseIter4MRFTrain)
            self.restoreMRF(sess,restore)
            self.initializeRemainingVars(sess)
            self.createCursors(sess)
            
            for step in range(self.mrfstartat,self.conf.mrf_training_iters+1):
                self.feed_dict[self.ph['keep_prob']] = 0.5
                self.doOpt(sess,step,self.conf.mrf_learning_rate)
                if step % self.conf.display_step == 0:
                    self.feed_dict[self.ph['keep_prob']] = 1.0
                    self.updateFeedDict(self.DBType.Train,sess=sess,distort=True)
                    train_loss = self.computeLoss(sess,[self.cost,basecost])
                    tt1 = self.computePredDist(sess,self.mrfPred)
                    tt2 = self.computePredDist(sess,self.basePred)
                    trainDist = [tt1.mean(),tt2.mean()]

                    numrep = int(self.conf.numTest/self.conf.batch_size)+1
                    val_loss = np.zeros([2,])
                    valDist = [0.,0.]
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val,sess=sess,distort=False)
                        val_loss += np.array(self.computeLoss(sess,[self.cost,basecost]))
                        tt1 = self.computePredDist(sess,self.mrfPred)
                        tt2 = self.computePredDist(sess,self.basePred)
                        valDist = [valDist[0]+tt1.mean(),valDist[1]+tt2.mean()]
                        
                    val_loss = val_loss/numrep
                    valDist = [valDist[0]/numrep,valDist[1]/numrep]
                    self.updateMRFLoss(step,train_loss,val_loss,trainDist,valDist)
                if step % self.conf.save_step == 0:
                    self.saveMRF(sess,step)
            print("Optimization Finished!")
            self.saveMRF(sess,step)
            
            
            
    def fineTrain(self, restore=True,trainPhase=True,trainType=0):
        self.createPH()
        self.createFeedDict()
        self.openDBs()
        self.feed_dict[self.ph['phase_train_fine']] = trainPhase
        self.feed_dict[self.ph['phase_train_base']] = False
        doBatchNorm = self.conf.doBatchNorm
        self.trainType = trainType
        
        with tf.variable_scope('base'):
            self.createBaseNetwork(doBatchNorm)
        with tf.variable_scope('mrf'):
            self.createMRFNetwork(doBatchNorm)
        with tf.variable_scope('fine'):
            self.createFineNetwork(doBatchNorm)

        self.createBaseSaver()
        self.createMRFSaver()
        self.createFineSaver()

        mod_labels = tf.maximum((self.ph['y']+1.)/2,0.01)
        # the labels shouldn't be zero because the prediction is an output of
        # exp. And it seems a lot of effort is wasted to make the prediction goto
        # zero rather than match the location.
        
        self.cost = tf.nn.l2_loss(self.finePred-tf.to_float(self.fine_labels))
        mrfcost = tf.nn.l2_loss(self.mrfPred-mod_labels)
        basecost =  tf.nn.l2_loss(self.basePred-self.ph['y'])
        self.createOptimizer()
        if self.conf.useMRF:
            predPair = [self.mrfPred,self.finePred]
        else:
            predPair = [self.basePred,self.finePred]
            
        with tf.Session() as sess:

            self.restoreBase(sess,True)
            self.restoreMRF(sess,True)
            self.restoreFine(sess,restore)
            self.initializeRemainingVars(sess)
            self.createCursors(sess)
            
            for step in range(self.finestartat,self.conf.fine_training_iters+1):
                self.doOpt(sess,step,self.conf.fine_learning_rate)
                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train,distort=True,sess=sess)
                    train_loss = self.computeLoss(sess,[self.cost,mrfcost,basecost])
                    tt1 = self.computePredDist(sess,self.basePred)
                    tt2 = self.computePredDist(sess,self.mrfPred)
                    tt3 = self.computeFinePredDist(sess,predPair)

                    trainDist = [tt3.mean(),tt2.mean(),tt1.mean()]

                    numrep = int(self.conf.numTest/self.conf.batch_size)+1
                    val_loss = np.zeros([3,])
                    valDist = [0.,0.,0.]
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val,distort=False,sess=sess)
                        val_loss += np.array(self.computeLoss(sess,[self.cost,mrfcost,basecost]))
                        tt1 = self.computePredDist(sess,self.basePred)
                        tt2 = self.computePredDist(sess,self.mrfPred)
                        tt3 = self.computeFinePredDist(sess,predPair)
                        valDist = [valDist[0]+tt3.mean(),
                                   valDist[1]+tt2.mean(),
                                   valDist[2]+tt1.mean()]
                        
                    val_loss = val_loss/numrep
                    valDist = [valDist[0]/numrep,valDist[1]/numrep,valDist[2]/numrep]
                    self.updateFineLoss(step,train_loss,val_loss,trainDist,valDist)
                if step % self.conf.save_step == 0:
                    self.saveFine(sess,step)
            print("Optimization Finished!")
            self.saveFine(sess,step)

    def jointTrain(self, restore=True):
        self.createPH()
        self.createFeedDict()
        self.openDBs()
        self.feed_dict[self.ph['phase_train_base']]=True
        self.feed_dict[self.ph['phase_train_fine']]=True
        doBatchNorm = self.conf.doBatchNorm
        
        with tf.variable_scope('base'):
            self.createBaseNetwork(doBatchNorm)
        with tf.variable_scope('mrf'):
            self.createMRFNetwork(doBatchNorm)
        with tf.variable_scope('fine'):
            self.createFineNetwork(doBatchNorm)

        self.createBaseSaver()
        self.createMRFSaver()
        self.createACSaver()
        self.createFineSaver()
        self.createJointSaver()

        mod_labels = tf.maximum((self.ph['y']+1.)/2,0.01)
        # the labels shouldn't be zero because the prediction is an output of
        # exp. And it seems a lot of effort is wasted to make the prediction goto
        # zero rather than match the location.
        
        finecost = tf.nn.l2_loss(self.finePred-self.fine_labels) 
        mrfcost = tf.nn.l2_loss(self.mrfPred-mod_labels)
        basecost =  tf.nn.l2_loss(self.basePred-self.ph['y'])
        self.cost = finecost + self.conf.joint_MRFweight*mrfcost
        
        self.createOptimizer()
        
        with self.env.begin() as txn,self.valenv.begin() as valtxn,tf.Session() as sess:

            self.restoreBase(sess,True)
            self.restoreMRF(sess,True)
            self.restoreFine(sess,True)
            self.restoreJoint(sess,restore)
            self.initializeRemainingVars(sess)
            self.createCursors(txn,valtxn)
            
            for step in range(self.jointstartat,self.conf.joint_training_iters+1):
                self.doOpt(sess,step,self.conf.joint_learning_rate)
                if step % self.conf.display_step == 0:
                    self.updateFeedDict(self.DBType.Train)
                    train_loss = self.computeLoss(sess,[self.cost,finecost,mrfcost,basecost])
                    numrep = int(self.conf.numTest/self.conf.batch_size)+1
                    val_loss = np.zeros([2,])
                    for rep in range(numrep):
                        self.updateFeedDict(self.DBType.Val)
                        val_loss += np.array(self.computeLoss(sess,[self.cost,finecost,mrfcost,basecost]))
                    val_loss = val_loss/numrep
                    self.updateJointLoss(step,train_loss,val_loss)
                if step % self.conf.save_step == 0:
                    self.saveJoint(sess,step)
            print("Optimization Finished!")
            self.saveJoint(sess,step)
                        

