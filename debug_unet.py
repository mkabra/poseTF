
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from stephenHeadConfig import conf
conf.cachedir = '/home/mayank/temp/cacheHead_Round2'

import PoseUNet
import tensorflow as tf
import PoseTools

conf.batch_size = 4
self = PoseUNet.PoseUNet(conf,'pose_unet')
val_dist, val_ims, val_preds, val_predlocs, val_locs = \
    self.classify_val(1)

sel = np.where(val_dist.sum(axis=1)>80)[0]
usel = np.where(val_dist.sum(axis=1)<20)[0]

sess = tf.InteractiveSession()
self.init_and_restore(sess,True,['loss','dist'])

##

ll = self.down_layers + self.up_layers

self.fd[self.ph['phase_train']] = False
bb = self.conf.batch_size

in1 = PoseTools.scale_images(val_ims[sel[:bb],...],2,conf)
self.fd[self.ph['x']] = in1
pp = sess.run(ll,self.fd)

in2 = PoseTools.scale_images(val_ims[usel[:bb],...],2,conf)
self.fd[self.ph['x']] = in2
qq = sess.run(ll,self.fd)

##
fig, ax = plt.subplots(2,7)

for ndx in range(7):
    dd1 = pp[ndx].flatten()
    dd2 = qq[ndx].flatten()
    ax[0,ndx].hist([dd1,dd2])

    dd1 = pp[-ndx-1].flatten()
    dd2 = qq[-ndx-1].flatten()
    ax[1,ndx].hist([dd1,dd2])
