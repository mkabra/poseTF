
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from stephenHeadConfig import conf
conf.cachedir = '/home/mayank/temp/cacheHead_Round2'

import PoseUNet
import tensorflow as tf
import PoseTools

# conf.batch_size = 4
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

##
fig,ax = plt.subplots(2,3)
ax = ax.flatten()
for ndx in range(30):
    curs = np.random.choice(sel)
    ax[0].cla()
    ax[0].imshow(val_ims[curs,...,0], cmap='gray')
    ax[0].scatter(val_locs[curs,:,0]*2,val_locs[curs,:,1]*2)
    for cls in range(5):
        ax[cls+1].cla()
        ax[cls+1].imshow(val_preds[curs,...,cls],vmax=1,vmin=-1)
        ax[cls+1].scatter(val_locs[curs,cls,0],val_locs[curs,cls,1])

    plt.pause(2)

