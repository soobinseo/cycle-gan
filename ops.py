import tensorflow as tf
import numpy as np

def lrelu(x, leak=0.2):

    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    output = f1 * x + f2 * abs(x)
    return output

def dataset_shuffling(x):
    shuffled_idx = np.arange(len(x))
    np.random.shuffle(shuffled_idx)
    return x[shuffled_idx, :]

def conv2d(tensor,
           output_dim,
           kernel_size,
           stride=1,
           activation_fn=tf.nn.relu,
           norm_fn=tf.contrib.layers.batch_norm,
           initializer=tf.truncated_normal_initializer(stddev=0.02),
           scope="name",
           reflect=False):
    with tf.variable_scope(scope):
        if reflect:
            tensor = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]])
            tensor_shape = tensor.get_shape().as_list()
            filter = tf.get_variable('filter', [kernel_size, kernel_size, tensor_shape[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(tensor, filter, strides=[1, stride, stride, 1], padding='VALID')
            if norm_fn is None:
                bn = conv
            else:
                bn = tf.contrib.layers.batch_norm(conv)
            if activation_fn is not None:
                output = activation_fn(bn)
            else:
                output = bn

            return output

            # return tf.contrib.layers.conv2d(re, num_outputs=output_dim, kernel_size=kernel_size,
            #                                 activation_fn=activation_fn, stride=stride,
            #                                 normalizer_fn=norm_fn, weights_initializer=initializer, scope=scope,
            #                                 padding="VALID")
        else:
            tensor_shape = tensor.get_shape().as_list()
            filter = tf.get_variable('filters', [kernel_size, kernel_size, tensor_shape[-1], output_dim], initializer=initializer)
            conv = tf.nn.conv2d(tensor, filter, strides=[1, stride, stride, 1], padding='SAME')
            bn = tf.contrib.layers.batch_norm(conv)
            if norm_fn is None:
                bn = conv
            else:
                bn = tf.contrib.layers.batch_norm(conv)
            if activation_fn is not None:
                output = activation_fn(bn)
            else:
                output = bn
            return output

def deconv2d(tensor, name, output_shape, kernel_size=3): # fractional-strided conv layer

    with tf.variable_scope(name):
        output_shape = [tf.shape(tensor)[0]] + output_shape
        filter = tf.get_variable('filters', [kernel_size, kernel_size, output_shape[-1], tensor.get_shape()[-1].value])
        deconv = tf.nn.conv2d_transpose(tensor, filter, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')
        bn = tf.contrib.layers.batch_norm(tf.reshape(deconv, output_shape))
        output = tf.nn.relu(bn)

        return output

def residual_block(tensor, name, kernel_size=3):

    output_dim = tensor.get_shape()[-1].value

    with tf.variable_scope(name):
        conv = conv2d(tensor, output_dim, kernel_size, reflect=True)
        conv2 = conv2d(conv, output_dim, kernel_size, activation_fn=None, reflect=True, scope="res2")

        return tensor + conv2


def bgr2rgb(bgr, shape=[256, 256, 3]):
    rgb = np.zeros(shape)
    rgb[:, :, 0] = bgr[:, :, 2]
    rgb[:, :, 1] = bgr[:, :, 1]
    rgb[:, :, 2] = bgr[:, :, 0]

    return rgb
