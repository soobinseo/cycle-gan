from ops import *

__author__="soobin3230"

class CycleGAN(object):

    def __init__(self, shape, lambda_=10., learning_rate=0.0001):
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        width, height, channel_A, channel_B = shape[0], shape[1], shape[2], shape[3]
        self.domain_A = tf.placeholder(tf.float32, [None, width, height, channel_A])
        self.domain_B = tf.placeholder(tf.float32, [None, width, height, channel_B])

        self.build_graph()

    def _generator(self, tensor, name, reuse=False):
        """
        generator for cycle-gan
        :param tensor: Input tensor
        :param name: Scope name
        :return: 3-D tensor with shape [width, height, channel]
        """

        width = tensor.get_shape()[1].value
        height = tensor.get_shape()[2].value

        with tf.variable_scope(name, reuse=reuse):

            # 9-blocks generator (6-blocks are available)
            c7s1 = conv2d(tensor, output_dim=32, kernel_size=7, stride=1, scope="c7s1")
            d64 = conv2d(c7s1, output_dim=64, kernel_size=3, stride=2, reflect=True, scope="d64")
            d128 = conv2d(d64, output_dim=128, kernel_size=3, stride=2, reflect=True, scope="d128")
            R_1 = residual_block(d128, name="Res_1")
            R_2 = residual_block(R_1, name="Res_2")
            R_3 = residual_block(R_2, name="Res_3")
            R_4 = residual_block(R_3, name="Res_4")
            R_5 = residual_block(R_4, name="Res_5")
            R_6 = residual_block(R_5, name="Res_6")
            R_7 = residual_block(R_6, name="Res_7")
            R_8 = residual_block(R_7, name="Res_8")
            R_9 = residual_block(R_8, name="Res_9")
            u64 = deconv2d(R_9, output_shape=[width//2, height//2, 64], name="u64")
            u32 = deconv2d(u64, output_shape=[width, height, 64], name="u32")
            output = conv2d(u32, output_dim=3, kernel_size=7, stride=1, norm_fn=None, activation_fn=tf.nn.tanh, reflect=True, scope="output_gen")

            return output

    def _discriminator(self, tensor, name, reuse=False):
        """

        :param tensor: Input tensor
        :param name: Scope name
        :return: Probability which is passed through sigmoid activation function
        """
        with tf.variable_scope(name, reuse=reuse):

            # This structure is called patch-GAN
            c64 = conv2d(tensor, output_dim=64, kernel_size=4, stride=1, activation_fn=lrelu, scope="c64", norm_fn=None)
            c128 = conv2d(c64, output_dim=128, kernel_size=4, stride=1, activation_fn=lrelu, scope="c128")
            c256 = conv2d(c128, output_dim=256, kernel_size=4, stride=1, activation_fn=lrelu, scope="c256")
            c512 = conv2d(c256, output_dim=512, kernel_size=4, stride=1, activation_fn=lrelu, scope="c512")
            output = conv2d(c512, output_dim=1, kernel_size=1, stride=1, activation_fn=tf.nn.sigmoid, scope="output_disc")

            return output

    def build_graph(self):
        """
        build network graph
        :return:
        """
        self.gen_AB = self._generator(self.domain_A, "generator_AB")
        self.gen_BA = self._generator(self.domain_B, "generator_BA")

        self.gen_BAB = self._generator(self.gen_BA, "generator_AB", reuse=True)
        self.gen_ABA = self._generator(self.gen_AB, "generator_BA", reuse=True)

        self.real_disc_A = self._discriminator(self.domain_A, "disciminator_A")
        self.fake_disc_A = self._discriminator(self.gen_BA, "discriminator_A", reuse=True)

        self.real_disc_B = self._discriminator(self.domain_B, "disciminator_B")
        self.fake_disc_B = self._discriminator(self.gen_AB, "discriminator_B", reuse=True)

        self.reconstruction_loss = self.lambda_ * (tf.reduce_mean(tf.abs((self.gen_ABA - self.domain_A))) + tf.reduce_mean(tf.abs((self.gen_BAB- self.domain_B))))

        self.disc_A_loss = -tf.reduce_mean(tf.log(self.real_disc_A) + tf.log(1.-self.fake_disc_A))
        self.disc_B_loss = -tf.reduce_mean(tf.log(self.real_disc_B) + tf.log(1.-self.fake_disc_B))

        self.gen_BA_loss = -self.disc_A_loss + self.reconstruction_loss
        self.gen_AB_loss = -self.disc_B_loss + self.reconstruction_loss

        self.gen_AB_train_op = tf.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.gen_AB_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_AB"))
        self.gen_BA_train_op = tf.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.gen_BA_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_BA"))
        self.disc_A_train_op = tf.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.disc_A_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_A"))
        self.disc_B_train_op = tf.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.disc_B_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_B"))

    def train_step(self):
        pass


