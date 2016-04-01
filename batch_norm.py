import tensorflow as tf
from tensorflow.python import control_flow_ops


def batch_norm(x, phase_train, scope='batch_norm'):
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
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
            name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
            name='gamma', trainable=True)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(tf.constant(phase_train),
            mean_var_with_update,
            lambda: (ema_mean, ema_var))

#        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
#            beta, gamma, 1e-3, affine)
#       above is is deprecated
        normed = tf.nn.batch_normalization(x, mean, var, 
	     beta, gamma, 1e-3)
    return normed
