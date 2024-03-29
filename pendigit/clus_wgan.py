import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self, x_dim=16):
        self.x_dim = x_dim
        self.name = 'pendigit/clus_wgan/d_net'

    def __call__(self, x, keep=1.0, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = tc.layers.fully_connected(
                x, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )

            fc1 = leaky_relu(tc.layers.batch_norm(fc1))
            
            fc2 = tc.layers.fully_connected(
                fc1, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )

            fc2 = leaky_relu(tc.layers.batch_norm(fc2))

            fc3 = tc.layers.fully_connected(fc2, 1,
                                            weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                            activation_fn=tf.identity
                                            )
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, z_dim=15, x_dim=16):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = 'pendigit/clus_wgan/g_net'

    def __call__(self, z, keep=1.0, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc1 = tcl.fully_connected(
                z, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            fc1 = leaky_relu(tc.layers.batch_norm(fc1))
             
            fc2 = tcl.fully_connected(
                fc1, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            fc2 = leaky_relu(tc.layers.batch_norm(fc2))

            fc3 = tc.layers.fully_connected(
                fc2, self.x_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder(object):
    def __init__(self, z_dim=15, dim_gen=5, x_dim=16):
        self.z_dim = z_dim
        self.dim_gen = dim_gen
        self.x_dim = x_dim
        self.name = 'pendigit/clus_wgan/enc_net'

    def __call__(self, x, keep=1.0, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = tc.layers.fully_connected(
                x, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            fc1 = leaky_relu(tc.layers.batch_norm(fc1))

            fc2 = tc.layers.fully_connected(
                fc1, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            fc2 = leaky_relu(tc.layers.batch_norm(fc2))

            fc3 = tc.layers.fully_connected(
                  fc2, self.z_dim, 
                  weights_initializer=tf.random_normal_initializer(stddev=0.02),
                  weights_regularizer=tc.layers.l2_regularizer(2.5e-5),                 
                  activation_fn=tf.identity)
            logits = fc3[:, self.dim_gen:]
            y = tf.nn.softmax(logits)
            return fc3[:, 0:self.dim_gen], y, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
