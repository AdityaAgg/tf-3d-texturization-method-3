import tensorflow as tf
import config
import util
from ops import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse

class Model(object):
    def __init__(self, batch_size_arg):
        self.nvx, self.npx, self.n_cls = config.shapenet_32_64()
        self.current_shapes = None
        self.nz = 100
        self.batch_size = batch_size_arg
        self.log_step = 50
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.nz])
        self.label = tf.placeholder(tf.float32, [self.batch_size, self.n_cls])
        self.color = tf.placeholder(tf.float32, [self.batch_size, self.npx, self.npx, 3])
        self.models_binvox = tf.placeholder(tf.float32, [self.batch_size, self.nvx, self.nvx, self.nvx, 1])
        self.models_colored_rgb = tf.placeholder(tf.float32, [self.batch_size, self.nvx, self.nvx, self.nvx, 3])
        self.train = tf.placeholder(tf.bool)
        self.init_ops(self.batch_size)
        self.encoder_called = False

    def train_model(self, sess, dataset, num_epochs):
        iters_per_epoch = dataset.num_examples // self.batch_size
        step = 0
        for epoch in range(1, num_epochs):
            for batch in range(iters_per_epoch):
                step += 1
                x, picture, geometry = dataset.next_batch(self.batch_size)
                noise = np.random.uniform(-1, 1, size=(self.batch_size, self.nz))
                dis_feed_dict = {self.z: noise, self.models_colored_rgb: x, self.color: picture,
                                 self.models_binvox: geometry, self.train: True}
                _, dis_loss = sess.run([self.D_opt, self.discrim_loss], feed_dict=dis_feed_dict)

                gen_feed_dict = {self.z: noise, self.models_binvox: geometry, self.train: True, self.color: picture}
                _, gen_loss = sess.run([self.G_opt, self.gen_loss], feed_dict=gen_feed_dict)
                # if step % self.log_step == 0:
                print('Iteration {0}: dis loss = {1:.4f}, gen loss = {2:.4f}'.format(step, dis_loss, gen_loss))

    def vizvox(self, gen_colors, gen_voxels, expected_colors):
        gen_voxels, expected_voxels = np.squeeze(gen_voxels)

        num_rows = int(len(gen_voxels)/8)
        num_cols = 8

        fig = plt.figure(figsize=(16, 8))
        outer = gridspec.GridSpec(num_rows, num_cols, wspace=0.2, hspace=0.2)

        for i in range(num_rows):
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
            j = 0

            ax = plt.Subplot(fig, inner[j], projection='3d')
            ax.set_title("Generated Voxels", y=1.05)
            ax.voxels(gen_voxels[i], facecolors=gen_colors[i])
            ax = plt.Subplot(fig, inner[j+1], projection='3d')
            ax.set_title("Expected Voxels", y=1.05)
            ax.voxels(gen_voxels[i], facecolors=expected_colors[i])

        plt.show()


    def generate_one_sample(self, dataset, sess):
        style, picture, geometry = dataset.get_random_sample()
        noise = np.random.uniform(-1, 1, size=(self.batch_size, self.nz))
        gen_feed_dict = {self.z: noise, self.models_binvox: geometry, self.train: True, self.color: picture}
        colors = np.asarray(sess.run([self.style_gen], feed_dict=gen_feed_dict))




        #visualization code

        #plot image
        #imgplot = plt.imshow(picture[0])
        #plt.show()



        #
        # #plot expected result
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.gca(projection='3d')
        # print "style max: ", np.amax(style)
        # print "style min: ", np.amin(style)
        # print "picture max: ", np.amax(picture)
        # print "picture min: ", np.amin(picture)

        color_vector = np.asarray(np.squeeze(style))
        color_vector[np.where(color_vector > 0.999)] = 1.0
        color_vector[np.where(color_vector < 0.0001)] = 0.0

        color_vector_2 = (np.asarray(np.squeeze(colors[0][0])).astype(np.float32) + 1.00001) * 0.499
        color_vector_2[np.where(color_vector > 0.999)] = 1.0
        color_vector_2[np.where(color_vector < 0.0001)] = 0.0
        self.vizvox(expected_colors=color_vector,  gen_voxels=geometry[0], gen_colors=color_vector_2)

        #ax.voxels(np.squeeze(geometry[0]), facecolors=color_vector, edgecolor='k')
        # plt.show()
        #
        #
        # #plot generated result
        # fig2 = plt.figure(figsize=(10, 10))
        # ax2 = fig2.gca(projection='3d')

        #
        # ax2.voxels(np.squeeze(geometry[0]), facecolors = color_vector, edgecolor='k')
        # plt.show()
        #
        # return geometry, colors
        #TODO: some code to go here

    def init_ops(self, batch_size=1):

        enc = Encoder()
        gen = Generator()
        dis = Discriminator()

        # encoders
        h4, h3, h2 = enc.color(self.color, self.train, self.n_cls)

        # generators

        self.style_gen = gen.style(self.models_binvox, h4, h3, h2, self.train)

        # discriminator on generated
        d_f= dis.discriminate(self.style_gen,h4, h3, h2, self.train, name='d_style')

        # discriminator on colored results
        # self.model_disc_results = dis.discriminate(self.models_colored_rgb, h4, h3, h2, self.train)
        d_r =dis.discriminate(self.models_colored_rgb, h4, h3, h2, self.train, name='d_style', reuse=True)

        # discriminator loss
        self.discrim_loss = tf.reduce_mean(sigmoid_ce_with_logits(d_r, tf.ones_like(d_r))) + tf.reduce_mean(
            sigmoid_ce_with_logits(d_f, tf.zeros_like(d_f)))

        # generator loss
        self.gen_loss = tf.reduce_mean(sigmoid_ce_with_logits(d_f, tf.ones_like(d_f)))

        # get vars
        t_vars = tf.trainable_variables()
        self.vars_G = [var for var in t_vars if var.name.startswith('g_')]
        self.vars_E = [var for var in t_vars if var.name.startswith('enc_')]
        self.vars_D = [var for var in t_vars if var.name.startswith('d_')]

        # Optimizers
        global_step = tf.Variable(0, trainable=False)
        learning_rate_decayed = tf.train.exponential_decay(0.0002, global_step, 9300, 0.5, staircase=False)
        self.G_opt = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.5).minimize(self.gen_loss,
                                                                                       var_list=self.vars_G + self.vars_E)
        self.D_opt = tf.train.AdamOptimizer(learning_rate_decayed, beta1=0.5).minimize(self.discrim_loss,
                                                                                       var_list=self.vars_D)






        # def update(self, color, edge, z, label):
        #     self.update_current_shapes(color,  z, label)
        #
        # def update_current_shapes(self, color,  z, label):
        #     feed_dict = {self.color: color,  self.z: z, self.label: label, self.train: False, self.models_binvox: ""} #TODO replace with binvox from shapenet
        #
        #     style = self.sess.run(self.style_gen, feed_dict=feed_dict)[0]
        #     voxel = voxel > 0.1
        #     style = np.clip(util.tanh2rgb(style), 0, 255)
        #     self.current_shapes = np.concatenate([self.models_binvox, style], -1).astype(np.uint8)


class Encoder(object):
    def color(self, color, train, n_cls, nc=3, nf=64, dropout=0.75, name="enc_color", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            # self.encoder_called =True
            c = conv2d(color, [4, 4, nc, nf], 'h1', bias=True)
            h1 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h1, [4, 4, nf, nf * 2], 'h2', bias=True)
            h2 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h2, [4, 4, nf * 2, nf * 4], 'h3', bias=True)
            h3 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h3, [4, 4, nf * 4, nf * 8], 'h4', bias=True)
            h4 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            f = tf.reshape(h4, [-1, 4 * 4 * nf * 8])
            y = linear(f, [4 * 4 * nf * 8, n_cls], 'h5', bias=True)
            return h4, h3, h2

    def edge(self, edge, train, n_cls, nc=1, nf=32, dropout=0.75, name="enc_edge", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            c = conv2d(edge, [4, 4, nc, nf], 'h1', bias=True)
            h1 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h1, [4, 4, nf, nf * 2], 'h2', bias=True)
            h2 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h2, [4, 4, nf * 2, nf * 4], 'h3', bias=True)
            h3 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h3, [4, 4, nf * 4, nf * 4], 'h4', bias=True)
            h4 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            f = tf.reshape(h4, [-1, 4 * 4 * nf * 4])
            y = linear(f, [4 * 4 * nf * 4, n_cls], 'h5', bias=True)
            return h4


class Generator(object):
    def style(self, voxel, h4, h3, h2, train, nc=3, nf=16, dropout=0.75, name="g_style", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            batch_size, _, _, _, _ = voxel.get_shape().as_list()
            _, _, _, nif4 = h4.get_shape().as_list()
            _, _, _, nif3 = h3.get_shape().as_list()
            _, _, _, nif2 = h2.get_shape().as_list()
            h4 = tf.tile(tf.expand_dims(h4, 1), [1, 4, 1, 1, 1])
            h3 = tf.tile(tf.expand_dims(h3, 1), [1, 8, 1, 1, 1])
            h2 = tf.tile(tf.expand_dims(h2, 1), [1, 16, 1, 1, 1])
            c = conv3d(voxel, [4, 4, 4, 1, nf], 'e1', bias=True, stride=1)
            e1 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))
            c = conv3d(e1, [4, 4, 4, nf, nf * 2], 'e2', bias=True)
            e2 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))
            c = conv3d(e2, [4, 4, 4, nf * 2, nf * 4], 'e3', bias=True)
            e3 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))
            c = conv3d(e3, [4, 4, 4, nf * 4, nf * 8], 'e4', bias=True)
            e4 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))
            c = deconv3d(tf.concat([e4, h4], 4), [4, 4, 4, nf * 8, nf * 8 + nif4], [batch_size, 8, 8, 8, nf * 8], 'd6',
                         bias=True)
            d6 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))
            c = conv3d(d6, [4, 4, 4, nf * 8, nf * 4], 'd5', bias=True, stride=1)
            d5 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))
            c = deconv3d(tf.concat([d5, h3], 4), [4, 4, 4, nf * 4, nf * 4 + nif3], [batch_size, 16, 16, 16, nf * 4],
                         'd4', bias=True)
            d4 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv3d(d4, [4, 4, 4, nf * 4, nf * 2], 'd3', bias=True, stride=1)
            d3 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = deconv3d(tf.concat([d3, h2], 4), [4, 4, 4, nf * 2, nf * 2 + nif2], [batch_size, 32, 32, 32, nf * 2],
                         'd2', bias=True)
            d2 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv3d(d2, [4, 4, 4, nf * 2, nc], 'd1', bias=True, stride=1)
            return tf.nn.tanh(c)


class Discriminator(object):
    def discriminate(self, x, h4, h3, h2, train, nc=3, nf = 16, dropout=0.75, name="d_style", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            batch_size, _, _, _, _ = x.get_shape().as_list()
            _, _, _, nif4 = h4.get_shape().as_list()
            _, _, _, nif3 = h3.get_shape().as_list()
            _, _, _, nif2 = h2.get_shape().as_list()

            h4 = tf.tile(tf.expand_dims(h4, 1), [1, 4, 1, 1, 1])
            h3 = tf.tile(tf.expand_dims(h3, 1), [1, 8, 1, 1, 1])
            h2 = tf.tile(tf.expand_dims(h2, 1), [1, 16, 1, 1, 1])

            # let us say some tanh inverse function
            #first conv_transpose
            dis_1 = conv3d(x, [4, 4, 4, nc, nf*2],  'dis_1', bias=True, stride=1) #turning off stride 1 on conv_transposes
            dis_1_post = tf.nn.dropout(lrelu(dis_1), keep_prob(dropout, train))

            #first conv
            dis_2 = conv3d(dis_1_post, [4, 4, 4,  nf * 2, nf * 2],'dis_2', bias=True)
            dis_2 = tf.concat([dis_2, h2], 4)
            dis_2_post = tf.nn.dropout(lrelu(dis_2), keep_prob(dropout, train))


            #second_conv_transpose
            dis_3 = conv3d(dis_2_post, [4, 4, 4, nf * 2 + nif2,  nf * 4], 'dis_3', bias=True, stride=1)
            dis_3_post = tf.nn.dropout(lrelu(batch_norm(dis_3, train, 'dis_3_bn3')), keep_prob(dropout, train))

            #second conv
            dis_4 = conv3d(dis_3_post, [4, 4, 4,  nf * 4, nf * 4], 'dis_4', bias=True)
            dis_4 = tf.concat([dis_4, h3], 4)
            dis_4_post = tf.nn.dropout(lrelu(dis_4), keep_prob(dropout, train))


            #third conv_transpose
            dis_5 = conv3d(dis_4, [4, 4, 4, nf * 4 + nif3, nf * 8], 'dis_5', bias=True, stride=1)
            dis_5_post = tf.nn.dropout(lrelu(batch_norm(dis_5, train, 'dis_5_bn5')), keep_prob(dropout, train))

            #third conv
            dis_6 = conv3d(dis_5_post, [4, 4, 4, nf * 8, nf * 8], 'dis_6', bias=True)
            dis_6 = tf.concat([dis_6, h4], 4)
            dis_6_post = tf.nn.dropout(lrelu(dis_6), keep_prob(dropout, train))

            #fourth conv_transpose
            dis_7 = conv3d(dis_6_post, [4, 4, 4, nf * 8 + nif4, nf * 8],'dis_7', bias=True, stride=1)
            dis_7_post = tf.nn.dropout(lrelu(batch_norm(dis_7, train, 'dis_7_bn7')), keep_prob(dropout, train))

            f = tf.reshape(dis_7_post, [-1, 4 * 4 * 4 * nf * 8])
            dis_label = linear(f, [4 * 4 * 4 * nf * 8, 1], 'dis_label', bias=True)
            return dis_label


# class Discriminator(object):
#     def __init__(self, n_cls, nz):
#         self.n_cls = n_cls
#         self.nz = nz
#         self.enc = Encoder()
#
#     def __call__(self, x, c, train, nf=16, name="d_style", reuse=False):
#         with tf.variable_scope(name, reuse=reuse):
#             shape = x.get_shape().as_list()
#             batch_size = shape[0]
#
#             if name == 'd_style':
#                 nc = 3
#             else:
#                 nc = 1
#
#             # add noise
#             x += tf.random_normal(shape)
#             c += tf.random_normal(c.get_shape())
#
#             # encode image
#             hc = self.enc.edge(c, train, self.n_cls, nc=nc, reuse=reuse)
#             hc = tf.reshape(hc, [batch_size, -1])
#
#             # encode voxels
#             u = conv3d(x, [4, 4, 4, nc, nf], 'h1', bias=True, stride=1)
#             hx = lrelu(u)
#
#             u = conv3d(hx, [4, 4, 4, nf, nf * 2], 'h2')
#             hx = lrelu(batch_norm(u, train, 'bn2'))
#
#             u = conv3d(hx, [4, 4, 4, nf * 2, nf * 4], 'h3')
#             hx = lrelu(batch_norm(u, train, 'bn3'))
#
#             u = conv3d(hx, [4, 4, 4, nf * 4, nf * 8], 'h4')
#             hx = lrelu(batch_norm(u, train, 'bn4'))
#
#             u = conv3d(hx, [4, 4, 4, nf * 8, nf * 16], 'h5')
#             hx = lrelu(batch_norm(u, train, 'bn5'))
#             hx = tf.reshape(hx, [batch_size, -1])
#
#             # discriminator
#             h = tf.concat([hc, hx], 1)
#             d = linear(h, [h.get_shape().as_list()[-1], 1], 'd', bias=True)
#
#             # classifier
#             y = linear(hx, [hx.get_shape().as_list()[-1], self.n_cls], 'y', bias=True)
#
#             # posterior
#             u = linear(hx, [hx.get_shape().as_list()[-1], self.nz], 'z', bias=True)
#             z = tf.nn.tanh(u)
#
#             return d, y, z
