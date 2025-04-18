from phi.tf.flow import *
from phi.tf.util import residual_block
import inspect, os


def projectnet2d_5x_8(velocity, training=False, trainable=True, reuse=None, scope="projectnet"):
    with tf.variable_scope(scope):
        y = velocity.staggered[:, :-1, :-1, :]
        downres_padding = sum([2**i for i in range(5)])
        y = tf.pad(y, [[0,0], [0,downres_padding], [0,downres_padding], [0,0]])
        resolutions = [y]
        for i, filters in enumerate([4, 8, 8, 8, 8]):
            y = tf.layers.conv2d(resolutions[0], filters, 2, strides=2, activation=tf.nn.relu, padding="valid",
                                 name="downconv_%d" % i, trainable=trainable, reuse=reuse)
            for j in range(2):
                y = residual_block(y, filters, name="downrb_%d_%d" % (i, j), training=training, trainable=trainable,
                                   reuse=reuse)
            resolutions.insert(0, y)

        for j, nb_channels in enumerate([8, 8]):
            y = residual_block(y, nb_channels, name="centerrb_%d" % j, training=training, trainable=trainable, reuse=reuse)

        for i, resolution_data in enumerate(resolutions[1:]):
            y = upsample2x(y)
            res_in = resolution_data[:, 0:y.shape[1], 0:y.shape[2], :]
            y = tf.concat([y, res_in], axis=-1)
            if i < len(resolutions) - 2:
                y = tf.pad(y, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC")
                y = tf.layers.conv2d(y, 8, 2, 1, activation=tf.nn.relu, padding="valid", name="upconv_%d" % i,
                                     trainable=trainable, reuse=reuse)
                for j, nb_channels in enumerate([8, 8]):
                    y = residual_block(y, nb_channels, 2, name="uprb_%d_%d" % (i, j), training=training,
                                       trainable=trainable, reuse=reuse)
            else:
                # Last iteration
                boundary_feature = tf.ones_like(y[...,0:1])
                boundary_feature = tf.pad(boundary_feature, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
                y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
                y = math.concat([y, boundary_feature], axis=-1)
                y = tf.layers.conv2d(y, 1, 2, 1, activation=None, padding="valid", name="upconv_%d" % i,
                                     trainable=trainable, reuse=reuse)

        velocity = StaggeredGrid(y).curl()
        path = os.path.join(os.path.dirname(inspect.getabsfile(projectnet2d_5x_8)), "projectnet2d_5x_8")
        return velocity, path