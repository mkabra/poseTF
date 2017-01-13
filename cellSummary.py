
# coding: utf-8

# In[18]:

import tensorflow as tf

efile = '/home/mayank/work/poseEstimation/cache/romainLegBottom/eval_train_summary/events.out.tfevents.1483696542.mayankWS'


# In[23]:


vals = []
names = []
first = True
for aa in tf.train.summary_iterator(efile):
    if not len(aa.summary.value):
        continue
    if first:
        for ndx,bb in enumerate(aa.summary.value):
            vals.append([])
            names.append(bb.tag)
        first = False
    for ndx,bb in enumerate(aa.summary.value):
        vals[ndx].append(bb.simple_value)


# In[30]:

for ndx in range(20):
    plt.figure()
    plt.plot(vals[ndx])
    plt.title(names[ndx])

