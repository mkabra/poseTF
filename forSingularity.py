import PoseUNet
import PoseUNetAttention
import tensorflow as tf
from poseConfig import aliceConfig as conf

# self = PoseUNetAttention.PoseUNetAttention(conf,unet_name='pose_unet_20180511',name='pose_unet_att_20180513')
# self.train_unet(False,0)
self = PoseUNet.PoseUNet(conf,name='pose_unet_20180511')
sess = self.init_net(0,True)
gg = self.classify_movie_trx('/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00020_CsChr_RigB_20150908T133237/movie.ufmf',
                             '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00020_CsChr_RigB_20150908T133237/registered_trx.mat',sess,1000,0)

# import PoseUNetAttention
# import tensorflow as tf
# from poseConfig import aliceConfig as conf
#
# self = PoseUNetAttention.PoseUNetAttention(conf,)