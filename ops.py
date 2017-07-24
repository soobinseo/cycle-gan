import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv2d(tensor,
           output_dim,
           kernel_size,
           stride=1,
           activation_fn=tf.nn.relu,
           norm_fn=tf.contrib.layers.batch_norm,
           initializer=tf.truncated_normal_initializer(stddev=0.02),
           scope="name",
           reflect=False):

    if reflect:
        re = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]])
        return tf.contrib.layers.conv2d(re, num_outputs=output_dim, kernel_size=kernel_size,
                                        activation_fn=activation_fn, stride=stride,
                                        normalizer_fn=norm_fn, weights_initializer=initializer, scope=scope,
                                        padding="VALID")
    else:
        return tf.contrib.layers.conv2d(tensor, num_outputs=output_dim, kernel_size=kernel_size,
                                        activation_fn=activation_fn, stride=stride,
                                        normalizer_fn=norm_fn, weights_initializer=initializer, scope=scope, padding="SAME")

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
        conv2 = conv2d(conv, output_dim, kernel_size, activation_fn=None, reflect=True)

        return tensor + conv2

