
# coding: utf-8

# In[ ]:

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def batch_norm(x, phase_train, scope='batch_norm'):
    # return tf.contrib.layers.batch_norm(x,center=True,scale=True,
    #                                     is_training=phase_train,
    #                                     scope=tf.get_variable_scope())

    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affine:      whether to affine-transform outputs
    Return:
        normed:      batch-normalized maps
    From: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177?noredirect=1#comment55758348_33950177
    """


    x_shape = tf.Tensor.get_shape(x)
    n_out = x_shape.as_list()[3]
    with tf.device(None):
        with tf.variable_scope(scope) as cursc:
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                name='gamma', trainable=True)

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):

                ema = tf.train.ExponentialMovingAverage(decay=0.9)

                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                mean, var = tf.cond(phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)),
                                    name='ema')
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
# Below is buggy - sigh.. MK Aug 9 2016            
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
#             def mean_var_with_update():
#                 with tf.control_dependencies([ema_apply_op]):
#                     return tf.identity(batch_mean), tf.identity(batch_var)
#             mean, var = control_flow_ops.cond(tf.constant(phase_train),
#                 mean_var_with_update,
#                 lambda: (ema_mean, ema_var))

#     #        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
#     #            beta, gamma, 1e-3, affine)
#     #       above is is deprecated
#             normed = tf.nn.batch_normalization(x, mean, var, 
#              beta, gamma, 1e-3)
#     return normed

def batch_norm_2D(x, phase_train, scope='batch_norm'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 2D BF input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
        affine:      whether to affine-transform outputs
    Return:
        normed:      batch-normalized maps
    From: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177?noredirect=1#comment55758348_33950177
    """
    x_shape = tf.Tensor.get_shape(x)
    n_out = x_shape.as_list()[1]
    with tf.device(None):
        with tf.variable_scope(scope):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                name='gamma', trainable=True)

            batch_mean, batch_var = tf.nn.moments(x, [0,], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed            

# why did this ever exist?    
# def batch_norm_2D(x, phase_train, scope='batch_norm'):
#     return x

