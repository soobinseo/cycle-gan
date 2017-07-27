from ops import *
import numpy as np
import matplotlib.pyplot as plt

__author__="soobin3230"

class CycleGAN(object):

    def __init__(self, shape, epoch=200, lambda_=10., learning_rate=0.0001):
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        width, height, channel_A, channel_B = shape[0], shape[1], shape[2], shape[3]
        self.domain_A = tf.placeholder(tf.float32, [None, width, height, channel_A])
        self.domain_B = tf.placeholder(tf.float32, [None, width, height, channel_B])
        self.epoch = epoch
        self.batch_size = 1
        self.build_graph()

    def _load_dataset(self):
        dataA = np.load("./data/dataset_trainA.npy")
        dataB = np.load("./data/dataset_trainB.npy")
        return dataA, dataB

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
            # R_2 = residual_block(R_1, name="Res_2")
            # R_3 = residual_block(R_2, name="Res_3")
            # R_4 = residual_block(R_3, name="Res_4")
            # R_5 = residual_block(R_4, name="Res_5")
            # R_6 = residual_block(R_5, name="Res_6")
            # R_7 = residual_block(R_6, name="Res_7")
            # R_8 = residual_block(R_7, name="Res_8")
            R_9 = residual_block(R_1, name="Res_9")
            u64 = deconv2d(R_9, output_shape=[width//2, height//2, 64], name="u64")
            u32 = deconv2d(u64, output_shape=[width, height, 64], name="u32")
            output = conv2d(u32, output_dim=3, kernel_size=7, stride=1, norm_fn=None, activation_fn=tf.nn.tanh, reflect=False, scope="output_gen")
            print output
            return output

    def _discriminator(self, tensor, name, reuse=False):
        """

        :param tensor: Input tensor
        :param name: Scope name
        :return: Probability which is passed through sigmoid activation function
        """
        with tf.variable_scope(name, reuse=reuse):

            # This structure is called patch-GAN
            c64 = conv2d(tensor, output_dim=64, kernel_size=4, stride=2, activation_fn=lrelu, scope="c64", norm_fn=None)
            c128 = conv2d(c64, output_dim=128, kernel_size=4, stride=2, activation_fn=lrelu, scope="c128")
            c256 = conv2d(c128, output_dim=256, kernel_size=4, stride=2, activation_fn=lrelu, scope="c256")
            c512 = conv2d(c256, output_dim=512, kernel_size=4, stride=1, activation_fn=lrelu, scope="c512", reflect=True)

            output = conv2d(c512, output_dim=1, kernel_size=4, stride=1, activation_fn=tf.nn.sigmoid, scope="output_disc", norm_fn=None, reflect=True)

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

        self.real_disc_A = self._discriminator(self.domain_A, "discriminator_A")
        print self.real_disc_A
        self.fake_disc_A = self._discriminator(self.gen_BA, "discriminator_A", reuse=True)
        self.real_disc_B = self._discriminator(self.domain_B, "discriminator_B")
        self.fake_disc_B = self._discriminator(self.gen_AB, "discriminator_B", reuse=True)

        self.reconstruction_loss = self.lambda_ * (tf.reduce_mean(tf.abs((self.gen_ABA - self.domain_A))) + tf.reduce_mean(tf.abs((self.gen_BAB- self.domain_B))))

        # Standard GAN loss
        # self.disc_A_loss = -tf.reduce_mean(tf.log(self.real_disc_A + 1e-5) + tf.log(1.-self.fake_disc_A + 1e-5))
        # self.disc_B_loss = -tf.reduce_mean(tf.log(self.real_disc_B + 1e-5) + tf.log(1.-self.fake_disc_B + 1e-5))
        # self.gen_BA_loss = -self.disc_A_loss + self.reconstruction_loss
        # self.gen_AB_loss = -self.disc_B_loss + self.reconstruction_loss

        # LSGAN loss
        self.reconstruction_loss = self.lambda_ * (tf.reduce_sum(tf.abs((self.gen_ABA - self.domain_A))) + tf.reduce_sum(tf.abs((self.gen_BAB- self.domain_B))))
        self.disc_A_loss = tf.reduce_sum(tf.square(1 - self.real_disc_A) + tf.square(self.fake_disc_A)) / 2
        self.disc_B_loss = tf.reduce_sum(tf.square(1 - self.real_disc_B) + tf.square(self.fake_disc_B)) / 2
        self.gen_BA_loss = tf.reduce_sum(tf.square(1 - self.fake_disc_A)) / 2 + self.reconstruction_loss
        self.gen_AB_loss = tf.reduce_sum(tf.square(1 - self.fake_disc_B)) / 2 + self.reconstruction_loss


        self.gen_AB_train_op = tf.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.gen_AB_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_AB"))
        self.gen_BA_train_op = tf.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.gen_BA_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_BA"))
        self.disc_A_train_op = tf.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.disc_A_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_A"))
        self.disc_B_train_op = tf.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.disc_B_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_B"))

    def train_step(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        dataA, dataB = self._load_dataset()
        dataA, dataB = dataA / 255, dataB / 255
        # trainable_variables = tf.trainable_variables()

        with tf.Session() as sess:
            sess.run(init)


            # print sess.run([trainable_variables])
            for i in range(self.epoch):
                dataA = dataset_shuffling(dataA)
                dataB = dataset_shuffling(dataB)
                batch_idxs = min(len(dataA), len(dataB)) // self.batch_size

                for idx in xrange(batch_idxs):
                    batch_A = dataA[idx * self.batch_size: (idx+1) * self.batch_size]
                    batch_B = dataB[idx * self.batch_size: (idx + 1) * self.batch_size]
                    sess.run([self.gen_AB_train_op, self.gen_BA_train_op, self.disc_A_train_op, self.disc_B_train_op],
                             feed_dict={self.domain_A:batch_A, self.domain_B:batch_B})

                    if idx % 50 == 0:
                        # print idx
                        print sess.run([self.disc_A_loss, self.disc_B_loss, self.gen_AB_loss, self.gen_BA_loss], feed_dict={self.domain_A:batch_A, self.domain_B:batch_B})


                img_AB = sess.run(self.gen_AB, feed_dict={self.domain_A: batch_A, self.domain_B:batch_B})
                img_BA = sess.run(self.gen_BA, feed_dict={self.domain_A: batch_A, self.domain_B:batch_B})

                plt.imsave("AB_%d.png" % i, img_AB[0])
                plt.imsave("BA_%d.png" % i, img_BA[0])

            saver.save(sess, "./result/model.ckpt")

if __name__ == '__main__':
    cyclegan = CycleGAN([256,256,3,3])
    cyclegan.train_step()
