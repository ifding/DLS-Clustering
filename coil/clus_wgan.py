import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self, x_dim = 49152):
        self.x_dim = x_dim
        self.name = 'coil/clus_wgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]

            x = tf.reshape(x, [bs, 128, 128, 3])
            conv1 = tc.layers.convolution2d(
                x, 64, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            conv3 = tc.layers.convolution2d(
                conv2, 256, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv3 = leaky_relu(conv3)
            conv4 = tc.layers.convolution2d(
                conv3, 512, [5, 5], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv4 = leaky_relu(conv4)

            conv5 = tc.layers.convolution2d(
                conv4, 1024, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv5 = leaky_relu(conv5)
            conv5 = tcl.flatten(conv5)

            #fc1 = tc.layers.fully_connected(
            #    conv4, 1024,
            #    weights_initializer=tf.random_normal_initializer(stddev=0.02),
            #    activation_fn=tf.identity
            #)
            #fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(conv5, 1, activation_fn=tf.identity)
            return fc2
            
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, z_dim = 40, x_dim = 49152):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = 'coil/clus_wgan/g_net'

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs: 
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(z)[0]
            fc1 = tc.layers.fully_connected(
                z, 8*8*1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            #fc1 = tc.layers.batch_norm(fc1)
            #fc1 = tf.nn.relu(fc1)
            #fc2 = tc.layers.fully_connected(
            #    fc1, 128 * 128 * 32,
            #    weights_initializer=tf.random_normal_initializer(stddev=0.02),
            #    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
            #    activation_fn=tf.identity
            #)
            fc1 = tf.reshape(fc1, tf.stack([bs, 8, 8, 1024]))
            fc1 = tc.layers.batch_norm(fc1)
            fc1 = tf.nn.relu(fc1)
            conv1 = tc.layers.convolution2d_transpose(
                fc1, 512, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = tc.layers.batch_norm(conv1)
            conv1 = tf.nn.relu(conv1)
            conv2 = tc.layers.convolution2d_transpose(
                conv1, 256, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv2 = tc.layers.batch_norm(conv2)
            conv2 = tf.nn.relu(conv2)
            conv3 = tc.layers.convolution2d_transpose(
                conv2, 128, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv3 = tc.layers.batch_norm(conv3)
            conv3 = tf.nn.relu(conv3)
            conv4 = tc.layers.convolution2d_transpose(
                conv3, 64, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv4 = tc.layers.batch_norm(conv4)
            conv4 = tf.nn.relu(conv4)
            conv5 = tc.layers.convolution2d_transpose(
                conv4, 3, [5, 5], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.tanh
            )
            conv5 = tf.reshape(conv5, tf.stack([bs, self.x_dim]))
            return conv5

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder(object):
    def __init__(self, z_dim = 40, dim_gen = 30, x_dim = 49152):
        self.z_dim = z_dim
        self.dim_gen = dim_gen
        self.x_dim = x_dim
        self.name = 'coil/clus_wgan/enc_net'

    def __call__(self, x, reuse=True):

        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 128, 128, 3])
            conv1 = tc.layers.convolution2d(
                x, 64, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            conv3 = tc.layers.convolution2d(
                conv2, 256, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv4 = leaky_relu(conv3)
                       
            conv4 = tc.layers.convolution2d(
                conv3, 512, [5, 5], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv4 = leaky_relu(conv4)

            conv5 = tc.layers.convolution2d(
                conv4, 1024, [5, 5], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv5 = leaky_relu(conv5)
            conv5 = tcl.flatten(conv5)
            #fc1 = tc.layers.fully_connected(
            #    conv4, 1024,
            #    weights_initializer=tf.random_normal_initializer(stddev=0.02),
            #    weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
            #    activation_fn=tf.identity
            #)
            #fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(conv5, self.z_dim, activation_fn=tf.identity)
            logits = fc2[:, self.dim_gen:]
            y = tf.nn.softmax(logits)
            return fc2[:, 0:self.dim_gen], y, logits


    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
