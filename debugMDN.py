import pickle
import os
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt


ind_loss = []
for ndx in range(conf.n_classes):
    ind_loss.append(inference.ind_loss[y[ndx]])
ind_loss = tf.transpose(tf.stack(ind_loss), [1, 0])

## training and test predictions

self.feed_dict[self.ph['phase_train_mdn']] = False

all_tt = []
all_locs = []
all_preds = []
for train_step in range(len(train_data)):
    data_ndx = train_step % len(m_train_data)
    cur_bpred = m_train_data[data_ndx][0]
    self.feed_dict[self.ph['base_locs']] = \
        PoseTools.getBasePredLocs(cur_bpred, self.conf)
    self.feed_dict[self.ph['base_pred']] = cur_bpred
    cur_te_loss, cur_ind_loss, pred_weights, pred_means, pred_std = \
        sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
                  self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)
    val_loss = cur_te_loss
    jj = np.argmax(pred_weights, axis=1)
    f_pred = np.zeros([16, 5, 2])
    for a_ndx in range(16):
        for b_ndx in range(5):
            f_pred[a_ndx, b_ndx, :] = pred_means[a_ndx, jj[a_ndx, b_ndx],
                                      b_ndx, :]

    tt1 = np.sqrt(((f_pred * 4 - self.feed_dict[self.ph['base_locs']]) ** 2).sum(axis=2))
    all_tt.append(tt1)
    all_locs.append(self.feed_dict[self.ph['base_locs']])
    all_preds.append([pred_weights, pred_means, pred_std, val_loss, cur_ind_loss])
    if train_step % 10 == 0:
        print('{},{}'.format(train_step, len(train_data)))

train_t = np.array(all_tt).reshape([-1, 5])
train_w = np.array([i[0] for i in all_preds]).reshape([-1, 256, 5])
train_m = np.array([i[1] for i in all_preds]).reshape([-1, 256, 5, 2])
train_s = np.array([i[2] for i in all_preds]).reshape([-1, 256, 5, 2])
train_l = np.array(all_locs).reshape([-1, 5, 2])
train_loss = np.array([i[4] for i in all_preds]).reshape([-1, 5])
train_lm = (train_l / 4) % 16

with open('mdn_train_preds','wb') as f:
    pickle.dump( [train_t,train_w,train_m,train_s,
                  train_l,train_loss,train_lm],f)


all_tt = []
all_locs = []
all_preds = []
for train_step in range(len(m_test_data) / 5):
    data_ndx = train_step % len(m_test_data)
    cur_bpred = m_test_data[data_ndx][0]
    self.feed_dict[self.ph['base_locs']] = \
        PoseTools.getBasePredLocs(cur_bpred, self.conf)
    self.feed_dict[self.ph['base_pred']] = cur_bpred
    cur_te_loss, cur_ind_loss, pred_weights, pred_means, pred_std = \
        sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
                  self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)
    val_loss = cur_te_loss
    jj = np.argmax(pred_weights, axis=1)
    f_pred = np.zeros([16, 5, 2])
    for a_ndx in range(16):
        for b_ndx in range(5):
            f_pred[a_ndx, b_ndx, :] = pred_means[a_ndx, jj[a_ndx, b_ndx],
                                      b_ndx, :]

    tt1 = np.sqrt(((f_pred * 4 - self.feed_dict[self.ph['base_locs']]) ** 2).sum(axis=2))
    all_tt.append(tt1)
    all_locs.append(self.feed_dict[self.ph['base_locs']])
    all_preds.append([pred_weights, pred_means, pred_std, val_loss, cur_ind_loss])
    if train_step % 10 == 0:
        print('{},{}'.format(train_step, len(m_test_data)))

all_t = np.array(all_tt).reshape([-1, 5])
all_w = np.array([i[0] for i in all_preds]).reshape([-1, 256, 5])
all_m = np.array([i[1] for i in all_preds]).reshape([-1, 256, 5, 2])
all_s = np.array([i[2] for i in all_preds]).reshape([-1, 256, 5, 2])
all_l = np.array(all_locs).reshape([-1, 5, 2])
all_loss = np.array([i[4] for i in all_preds]).reshape([-1, 5])
all_lm = (all_l / 4) % 16

with open('mdn_test_preds','wb') as f:
    pickle.dump( [all_t,all_w,all_m,all_s,all_l,all_loss,all_lm],f)

sel_count = -1

##

with open('mdn_test_preds','rb') as f:
    [all_t, all_w, all_m, all_s, all_l, all_loss, all_lm] = pickle.load(f)

with open('mdn_train_preds','rb') as f:
    [train_t, train_w, train_m, train_s,
     train_l, train_loss, train_lm] = pickle.load(f)

##

sel_count += 1
zz = np.where(all_t.sum(axis=1) > 120)[0]

kk = np.arange(256) // 4
xx = kk // 8
yy = kk % 8

while True:
    # sel = np.random.choice(zz)
    sel = zz[sel_count]
    pp = np.where(all_t[sel, :] > 40)[0]
    if pp.size > 0:
        break
    else:
        sel_count += 1
sel_cls = np.random.choice(pp)
d_ndx = sel // conf.batch_size
i_ndx = sel % conf.batch_size

cur_l = all_l[sel, sel_cls, :] / 4 // 16

sel_ndx = np.where((xx == (cur_l[0])) & (yy == (cur_l[1])))[0]

diff_m = all_m[sel, sel_ndx, sel_cls, :] * 4 - all_l[sel, sel_cls, :]

mdn_pred_out = np.zeros([128, 128])
cur_m = all_m[sel, :, sel_cls, :]
cur_s = all_s[sel, :, sel_cls, :]
cur_w = all_w[sel, :, sel_cls]
for ndx in range(256):
    osz = 128
    if cur_w[ndx] < 0.02:
        continue
    cur_locs = cur_m[ndx:ndx + 1, :].astype('int')[np.newaxis, :, :]
    cur_scale = cur_s[ndx, :].mean().astype('int')
    cur_limg = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
    mdn_pred_out += cur_w[ndx] * cur_limg[0, ..., 0]

f, ax = plt.subplots(1, 3)
c_im = (m_test_data[d_ndx][0][i_ndx, :, :, sel_cls] + 1) / 2
c_im = np.clip(c_im, 0, 1)
i_im = m_test_data[d_ndx][3][i_ndx, 0, :, :]
ax[0].imshow(c_im, interpolation='nearest')
ax[0].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4,s = 6)
ax[0].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r', s = 5)
ax[1].imshow(mdn_pred_out, vmax=1., interpolation='nearest')
ax[1].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4, s= 5)
ax[1].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r', s=6)
ax[0].set_title('{:.2f},{},{}'.format(c_im.max(), sel_cls, sel))
ax[2].imshow(i_im, cmap='gray')
ax[2].scatter(all_l[sel, sel_cls, 0], all_l[sel, sel_cls, 1])
# ax[2].scatter(all_m[sel,sel_ndx,sel_cls,0]*4, all_m[sel,sel_ndx,sel_cls,1]*4,c='r')

for cur_ax in ax:
    cur_ax.set_xticks(np.arange(0, 128, 16));
    cur_ax.set_yticks(np.arange(0, 128, 16));

    cur_ax.set_xticklabels(np.arange(0, 128, 16));
    cur_ax.set_yticklabels(np.arange(0, 128, 16));
    # Gridlines based on minor ticks
    cur_ax.grid(which='major', color='k', linestyle='-', linewidth=0.1)

##
print('yes')

## ------------------------------------------------------ ##


#
# zz = np.where(all_t.sum(axis=1) > 40)[0]
#
# m_val = []
# for sel in zz:
#     sel_cls = np.argmax(all_t[sel, :])
#     d_ndx = sel // conf.batch_size
#     i_ndx = sel % conf.batch_size
#     c_im = m_train_data[d_ndx][0][i_ndx, :, :, sel_cls]
#     m_val.append(c_im.max())
#
# plt.scatter(all_t.sum(axis=1)[zz], m_val)
#
# ## select a training example with bad error
#
# zz = np.where(all_t.sum(axis=1) > 40)[0]
#
# kk = np.arange(256) // 4
# xx = kk // 8
# yy = kk % 8
#
# while True:
#     # sel = np.random.choice(zz)
#     sel = zz[sel_count]
#     pp = np.where(all_t[sel, :] > 20)[0]
#     if pp.size > 0:
#         break
#     else:
#         sel_count += 1
# sel_cls = np.random.choice(pp)
# d_ndx = sel // conf.batch_size
# i_ndx = sel % conf.batch_size
#
# cur_l = all_l[sel, sel_cls, :] / 4 // 16
#
# sel_ndx = np.where((xx == (cur_l[0])) & (yy == (cur_l[1])))[0]
#
# diff_m = all_m[sel, sel_ndx, sel_cls, :] * 4 - all_l[sel, sel_cls, :]
#
# mdn_pred_out = np.zeros([128, 128])
# cur_m = all_m[sel, :, sel_cls, :]
# cur_s = all_s[sel, :, sel_cls, :]
# cur_w = all_w[sel, :, sel_cls]
# for ndx in range(256):
#     osz = 128
#     if cur_w[ndx] < 0.02:
#         continue
#     cur_locs = cur_m[ndx:ndx + 1, :].astype('int')[np.newaxis, :, :]
#     cur_scale = cur_s[ndx, :].mean().astype('int')
#     cur_limg = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
#     mdn_pred_out += cur_w[ndx] * cur_limg[0, ..., 0]
#
# f, ax = plt.subplots(1, 3)
# c_im = (m_train_data[d_ndx][0][i_ndx, :, :, sel_cls] + 1) / 2
# c_im = np.clip(c_im, 0, 1)
# i_im = m_train_data[d_ndx][3][i_ndx, 0, :, :]
# ax[0].imshow(c_im, interpolation='nearest')
# ax[0].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
# ax[0].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r')
# ax[1].imshow(mdn_pred_out, vmax=1., interpolation='nearest')
# ax[1].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
# ax[1].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r')
# ax[0].set_title('{:.2f},{},{}'.format(c_im.max(), sel_cls, sel))
# ax[2].imshow(i_im, cmap='gray')
# ax[2].scatter(all_l[sel, sel_cls, 0], all_l[sel, sel_cls, 1])
# # ax[2].scatter(all_m[sel,sel_ndx,sel_cls,0]*4, all_m[sel,sel_ndx,sel_cls,1]*4,c='r')
#
# for cur_ax in ax:
#     cur_ax.set_xticks(np.arange(0, 128, 16));
#     cur_ax.set_yticks(np.arange(0, 128, 16));
#
#     cur_ax.set_xticklabels(np.arange(0, 128, 16));
#     cur_ax.set_yticklabels(np.arange(0, 128, 16));
#     # Gridlines based on minor ticks
#     cur_ax.grid(which='major', color='k', linestyle='-', linewidth=0.1)
#
# sel_count += 1
##

# vv = tf.global_variables()
# gg = tf.gradients(self.loss, vv)
# kk = [ndx for ndx, i in enumerate(gg) if i is not None]
# vv = [vv[i] for i in kk]
# gg = [gg[i] for i in kk]

d_ndx = sel // conf.batch_size
i_ndx = sel % conf.batch_size
cur_in_img = np.tile(m_test_data[d_ndx][0][i_ndx, ...],
                     [conf.batch_size, 1, 1, 1])
self.feed_dict[self.ph['base_pred']] = cur_in_img
self.feed_dict[self.ph['base_locs']] = \
    PoseTools.getBasePredLocs(cur_in_img, self.conf)
self.feed_dict[self.ph['step']] = 1000
self.feed_dict[self.ph['phase_train_mdn']] = True

for count in range(1):
    sess.run(self.opt, self.feed_dict)

p_loss, p_ind_loss, p_w, p_m, p_s = \
    sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
              self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)

mdn_p_img = np.zeros([128, 128])
for m_ndx in range(p_m.shape[1]):
    if p_w[0, m_ndx, sel_cls] < 0.02:
        continue
    cur_locs = p_m[0:1, m_ndx:m_ndx + 1, sel_cls, :].astype('int')
    cur_scale = 3
    curl = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
    mdn_p_img += p_w[0, m_ndx, sel_cls] * curl[0, ..., 0]


f, ax = plt.subplots(1, 4)
c_im = (m_test_data[d_ndx][0][i_ndx, :, :, sel_cls] + 1) / 2
c_im = np.clip(c_im, 0, 1)
i_im = m_test_data[d_ndx][3][i_ndx, 0, :, :]
ax[0].imshow(c_im, interpolation='nearest')
ax[0].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
ax[0].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r',s=5)
ax[1].imshow(mdn_pred_out, vmax=1., interpolation='nearest')
ax[1].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4,s=5)
ax[1].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r', s=5)
ax[0].set_title('{:.2f},{},{}'.format(c_im.max(), sel_cls, sel))
ax[2].imshow(mdn_p_img, vmax=1., interpolation='nearest')
ax[2].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4, s = 5)
ax[3].imshow(i_im, cmap='gray')
ax[3].scatter(all_l[sel, sel_cls, 0], all_l[sel, sel_cls, 1],s=5)
# ax[2].scatter(all_m[sel,sel_ndx,sel_cls,0]*4, all_m[sel,sel_ndx,sel_cls,1]*4,c='r')

for cur_ax in ax:
    cur_ax.set_xticks(np.arange(0, 128, 16));
    cur_ax.set_yticks(np.arange(0, 128, 16));

    cur_ax.set_xticklabels(np.arange(0, 128, 16));
    cur_ax.set_yticklabels(np.arange(0, 128, 16));
    # Gridlines based on minor ticks
    cur_ax.grid(which='major', color='k', linestyle='-', linewidth=0.1)


##
mod_tt = []
mod_locs = []
mod_preds = []
for train_step in range(len(train_data)):
    data_ndx = train_step % len(m_train_data)
    cur_bpred = m_train_data[data_ndx][0]
    self.feed_dict[self.ph['base_locs']] = \
        PoseTools.getBasePredLocs(cur_bpred, self.conf)
    self.feed_dict[self.ph['base_pred']] = cur_bpred
    cur_te_loss, cur_ind_loss, pred_weights, pred_means, pred_std = \
        sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
                  self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)
    val_loss = cur_te_loss
    jj = np.argmax(pred_weights, axis=1)
    f_pred = np.zeros([16, 5, 2])
    for a_ndx in range(16):
        for b_ndx in range(5):
            f_pred[a_ndx, b_ndx, :] = pred_means[a_ndx, jj[a_ndx, b_ndx],
                                      b_ndx, :]

    tt1 = np.sqrt(((f_pred * 4 - self.feed_dict[self.ph['base_locs']]) ** 2).sum(axis=2))
    mod_tt.append(tt1)
    mod_locs.append(self.feed_dict[self.ph['base_locs']])
    mod_preds.append([pred_weights, pred_means, pred_std, val_loss, cur_ind_loss])
    if train_step % 10 == 0:
        print('{},{}'.format(train_step, len(train_data)))

mod_t = np.array(mod_tt).reshape([-1, 5])
mod_w = np.array([i[0] for i in mod_preds]).reshape([-1, 256, 5])
mod_m = np.array([i[1] for i in mod_preds]).reshape([-1, 256, 5, 2])
mod_s = np.array([i[2] for i in mod_preds]).reshape([-1, 256, 5, 2])
mod_l = np.array(mod_locs).reshape([-1, 5, 2])
mod_loss = np.array([i[4] for i in mod_preds]).reshape([-1, 5])
mod_lm = (mod_l / 4) % 16

##

hh = train_loss - mod_loss
qq = np.flipud(np.argsort(hh[:, sel_cls]))

f, ax = plt.subplots(2, 5)
ax = ax.flatten()

for ndx in range(10):
    cur_ndx = qq[ndx]
    cur_d_ndx = cur_ndx // conf.batch_size
    cur_i_ndx = cur_ndx % conf.batch_size
    mdn_pred_out1 = np.zeros([128, 128])
    mdn_pred_out2 = np.zeros([128, 128])

    for m_ndx in range(mod_m.shape[1]):
        if mod_w[cur_ndx, m_ndx, sel_cls] < 0.02:
            continue
        cur_locs = mod_m[cur_ndx:cur_ndx + 1, m_ndx:m_ndx + 1, sel_cls, :].astype('int')
        cur_scale = 3
        curl = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
        mdn_pred_out1 += mod_w[cur_ndx, m_ndx, sel_cls] * curl[0, ..., 0]

    for m_ndx in range(train_m.shape[1]):
        if train_w[cur_ndx, m_ndx, sel_cls] < 0.02:
            continue
        cur_locs = train_m[cur_ndx:cur_ndx + 1, m_ndx:m_ndx + 1, sel_cls, :].astype('int')
        cur_scale = 3
        curl = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
        mdn_pred_out2 += train_w[cur_ndx, m_ndx, sel_cls] * curl[0, ..., 0]
    mdn_pred_out = np.zeros([128, 128, 3])
    mdn_pred_out[:, :, 0] = mdn_pred_out1
    mdn_pred_out[:, :, 1] = mdn_pred_out2
    cur_bpred = (m_train_data[cur_d_ndx][0][cur_i_ndx, :, :, sel_cls] + 1) / 2
    mdn_pred_out[:, :, 2] = cur_bpred
    mdn_pred_out = np.clip(mdn_pred_out, 0, 1)
    ax[ndx].imshow(mdn_pred_out, interpolation='nearest')
