import numpy, os
import tensorflow as tf
from phi.tf.util import *
from phi.control.sequences import *
from phi.control.nets.force.forcenets import forcenet2d_3x_16 as forcenet



def ik(initial_density, target_density, trainable=False):
    # conv = conv_function("model", "model/smokeik/sim_000301/checkpoint_00014802/model.ckpt")  # 64x64
    # conv = conv_function("model", "model/smokeik/sim_000430/checkpoint_00005747/model.ckpt")  # 128x128
    with tf.variable_scope("ik"):
        vec_pot = ik_resnet(initial_density, target_density, trainable=trainable, training=False, reuse=tf.AUTO_REUSE)
    with tf.variable_scope("curl"):
        velocity = vec_pot.curl()
    velocity = velocity.pad(0, 1, "symmetric")
    return velocity, vec_pot


def ik_resnet(initial_density, target_density, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    y = tf.concat([initial_density, target_density], axis=-1)
    y = tf.pad(y, [[0,0], [0,1+2+4+4], [0,1+2+4+4], [0,0]])
    resolutions = [ y ]
    for i in range(1,4): # 1/2, 1/4, 1/8
        y = tf.layers.conv2d(resolutions[0], 16, 2, strides=2, activation=tf.nn.relu, padding="valid", name="downconv_%d"%i, trainable=trainable, reuse=reuse)
        for j, nb_channels in enumerate([16, 16, 16]):
            y = residual_block(y, nb_channels, name="downrb_%d_%d" % (i,j), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)

    y = tf.layers.conv2d(y, 16, 2, 1, activation=tf.nn.relu, padding="valid", name="centerconv_1", trainable=trainable, reuse=reuse)
    for j, nb_channels in enumerate([16, 16, 16]):
        y = residual_block(y, nb_channels, name="centerrb_%d" % j, training=training, trainable=trainable, reuse=reuse)

    for i in range(1, len(resolutions)):
        y = upsample2x(y)
        res_in = resolutions[i][:, 0:y.shape[1], 0:y.shape[2], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions)-1:
            y = tf.pad(y, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC")
            y = tf.layers.conv2d(y, 16, 2, 1, activation=tf.nn.relu, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse)
            for j, nb_channels in enumerate([16, 16, 16]):
                y = residual_block(y, nb_channels, 2, name="uprb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(y, [[0,0], [0,1], [0,1], [0,0]], mode="SYMMETRIC")
            y = tf.layers.conv2d(y, 1, 2, 1, activation=None, padding="valid", name="upconv_%d"%i, trainable=trainable, reuse=reuse)

    return StaggeredGrid(y) # This is the vector potential


def sm_resnet(initial_density, target_density, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    y = tf.concat([initial_density, target_density], axis=-1)
    downres_padding = sum([2**i for i in range(5)])
    y = tf.pad(y, [[0,0], [0,downres_padding], [0,downres_padding], [0,0]])
    resolutions = [ y ]
    for i, filters in enumerate([4, 8, 16, 16, 16]):
        y = tf.layers.conv2d(resolutions[0], filters, 2, strides=2, activation=tf.nn.relu, padding="valid", name="downconv_%d"%i, trainable=trainable, reuse=reuse)
        for j in range(2):
            y = residual_block(y, filters, name="downrb_%d_%d" % (i,j), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)

    for j, nb_channels in enumerate([16, 16, 16]):
        y = residual_block(y, nb_channels, name="centerrb_%d" % j, training=training, trainable=trainable, reuse=reuse)

    for i, resolution_data in enumerate(resolutions[1:]):
        y = upsample2x(y)
        res_in = resolution_data[:, 0:y.shape[1], 0:y.shape[2], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions)-2:
            y = tf.pad(y, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC")
            y = tf.layers.conv2d(y, 16, 2, 1, activation=tf.nn.relu, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse)
            for j, nb_channels in enumerate([16, 16]):
                y = residual_block(y, nb_channels, 2, name="uprb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(y, [[0,0], [0,1], [0,1], [0,0]], mode="SYMMETRIC")
            y = tf.layers.conv2d(y, 1, 2, 1, activation=None, padding="valid", name="upconv_%d"%i, trainable=trainable, reuse=reuse)

    return y


class GraphBuilder(PartitioningExecutor):

    def __init__(self, sim, true_densities, trainable_n, info, force_inference, ik_trainable=False):
        self.sim = sim
        self.true_densities = true_densities
        self.trainable_n = trainable_n
        self.info = info
        self.force_inference = force_inference
        self.ik_trainable = ik_trainable

    def create_frame(self, index, step_count):
        frame = PartitioningExecutor.create_frame(self, index, step_count)
        frame.true = self.true_densities[index]
        frame.pred = []
        frame.real = None
        frame.force = None
        frame.prev_force = None
        frame.jerk = None
        frame.density = None
        frame.velocity = None
        frame.prev_velocity = None
        if index == 0:
            frame.pred = [ frame.true ]
            frame.real = frame.true
            frame.density = frame.true
        elif index == step_count:
            frame.pred = [ frame.true ]
            frame.density = frame.true
        return frame

    def run_sm(self, n, initial_density, target_density):
        with tf.variable_scope("sm%d" % n):
            return sm_resnet(initial_density, target_density, trainable=n in self.trainable_n)

    def run_ik(self, initial_density, target_density):
        return ik(initial_density, target_density, trainable=self.ik_trainable)

    def run_advect(self, velocity, density):
        return velocity.advect(density)

    def run_force(self, initial_velocity, target_velocity, initial_density, real_target_density):
        if self.force_inference == "forcenet":
            force, self.forcenet_path = forcenet(initial_density, initial_velocity, target_velocity)
        else:
            next_velocity = initial_velocity.advect(initial_velocity) + self.sim.buoyancy(real_target_density)
            if self.force_inference == "exact":
                next_velocity = self.sim.divergence_free(next_velocity)
            force = target_velocity - next_velocity
        return force

    def run_jerk(self, initial_velocity, initial_force, next_force):
        advected_force = initial_velocity.advect(initial_force)
        return next_force - advected_force

    def partition(self, n, initial_frame, target_frame, center_frame):
        PartitioningExecutor.partition(self, n, initial_frame, target_frame, center_frame)
        center_frame.density = self.run_sm(n, initial_frame.density, target_frame.density)
        center_frame.pred.append(center_frame.density)

    def execute_step(self, initial_frame, target_frame):
        PartitioningExecutor.execute_step(self, initial_frame, target_frame)
        initial_frame.velocity, initial_frame.vec_pot = target_frame.prev_velocity, _ = self.run_ik(initial_frame.real, target_frame.pred[-1])
        target_frame.real = target_frame.density = self.run_advect(initial_frame.velocity, initial_frame.real)
        if initial_frame.prev_velocity is not None:
            initial_frame.force = self.run_force(initial_frame.prev_velocity, initial_frame.velocity, initial_frame.real, target_frame.real)
            target_frame.prev_force = initial_frame.force
        if initial_frame.prev_force is not None:
            initial_frame.jerk = self.run_jerk(initial_frame.prev_velocity, initial_frame.prev_force, initial_frame.force)

    def load_checkpoints(self, max_n, checkpoint_dict, preload_n):
        # Force
        if self.force_inference == "forcenet":
            self.info("Loading ForceNet checkpoint from %s..." % self.forcenet_path)
            self.sim.restore(self.forcenet_path, scope="ForceNet")
        # IK
        ik_checkpoint = os.path.expanduser(checkpoint_dict["IK"])
        self.info("Loading IK checkpoint from %s..." % ik_checkpoint)
        self.sim.restore(ik_checkpoint, scope="ik")
        # SM
        n = 2
        while n <= max_n:
            if n == max_n and not preload_n: return
            checkpoint_path = None
            i = n
            while not checkpoint_path:
                if "SM%d"%i in checkpoint_dict:
                    checkpoint_path = os.path.expanduser(checkpoint_dict["SM%d"%i])
                else:
                    i //= 2
            if i == n:
                self.info("Loading SM%d checkpoint from %s..." % (n, checkpoint_path))
                self.sim.restore(checkpoint_path, scope="sm%d" % n)
            else:
                self.info("Loading SM%d weights from SM%d checkpoint from %s..." % (n, i, checkpoint_path))
                self.sim.restore_new_scope(checkpoint_path, "sm%d" % i, "sm%d" % n)
            n *= 2


    def load_all_from(self, max_n, ik_checkpoint, sm_checkpoint, sm_n):
        # IK
        self.info("Loading IK checkpoint from %s..." % ik_checkpoint)
        self.sim.restore(ik_checkpoint, scope="ik")
        # SM
        n = 2
        while n <= max_n:
            source_n = sm_n(n) if callable(sm_n) else sm_n
            self.info("Loading SM%d weights from SM%d checkpoint from %s..." % (n, source_n, sm_checkpoint))
            self.sim.restore_new_scope(sm_checkpoint, "sm%d" % source_n, "sm%d" % n)
            n *= 2

    def lookup(self, array):
        return array



class EagerExecutor(GraphBuilder):

    def __init__(self, sim, true_densities, info, force_inference):
        GraphBuilder.__init__(self, sim, true_densities, [], info, force_inference)
        self.initial_density = self.sim.placeholder(name="initial_density")
        self.target_density = self.sim.placeholder(name="target_density")
        self.initial_velocity = self.sim.placeholder("velocity", name="initial_velocity")
        self.target_velocity = self.sim.placeholder("velocity", name="target_velocity")
        info("Building IK graph...")
        self.ik_out_velocity, self.ik_out_vec_pot = ik(self.initial_density, self.target_density)

        info("Building force graph...")
        self.force_out, self.forcenet_path = forcenet(self.initial_density, self.initial_velocity, self.target_velocity)

        self.sm_out_by_n = {}
        n = 2
        while n <= len(true_densities):
            self.info("Building SM%d graph..."%n)
            self.sm_out_by_n[n] = GraphBuilder.run_sm(self, n, self.initial_density, self.target_density)
            n *= 2
        self.feed = {}

    def lookup(self, array):
        if isinstance(array, numpy.ndarray):
            return array
        else:
            return self.feed[array]

    def run_ik(self, initial_density, target_density):
        self.feed[self.initial_density] = self.lookup(initial_density)
        self.feed[self.target_density] = self.lookup(target_density)
        result = self.sim.run([self.ik_out_velocity, self.ik_out_vec_pot], feed_dict=self.feed)
        return result

    def run_sm(self, n, initial_density, target_density):
        self.feed[self.initial_density] = self.lookup(initial_density)
        self.feed[self.target_density] = self.lookup(target_density)
        result = self.sim.run(self.sm_out_by_n[n], feed_dict=self.feed)
        return result

    def run_advect(self, velocity, density):
        if not isinstance(density, numpy.ndarray):
            density = self.feed[density]
        if not isinstance(velocity.staggered, numpy.ndarray):
            velocity = self.feed[velocity]
        result = velocity.advect(density)
        return result

    # def run_force(self, initial_velocity, target_velocity, initial_density, real_target_density):
        # self.feed[self.initial_density] = self.lookup(initial_density)
        # self.feed[self.initial_velocity] = self.lookup(initial_velocity)
        # self.feed[self.target_velocity] = self.lookup(target_velocity)
        # return self.sim.run(self.force_out, feed_dict=self.feed)


    def set_dict(self, feed_dict):
        self.feed = feed_dict



def get_divide_strategy(name):
    if name == "adaptive":
        return AdaptivePlanSequence
    elif name == "binary":
        return TreeSequence
    else:
        raise ValueError("unknown divide strategy: %s" % name)


class MultiShapeEagerExecutor(EagerExecutor):

    def __init__(self, sim, true_densities, info, force_inference):
        GraphBuilder.__init__(self, sim, true_densities, [], info, force_inference)
        self.initial_density = self.sim.placeholder(name="initial_density")
        self.target_density = self.sim.placeholder(name="target_density")
        self.initial_velocity = self.sim.placeholder("velocity", name="initial_velocity")
        self.target_velocity = self.sim.placeholder("velocity", name="target_velocity")
        info("Building single-batch IK graph...")
        initial_sum = math.expand_dims(math.sum(self.initial_density, axis=0), axis=0)
        target_sum = math.expand_dims(math.sum(self.target_density, axis=0), axis=0)
        ik_single_batch = ik(initial_sum, target_sum)
        ik_single_batch_vel = ik_single_batch[0].staggered
        ik_single_batch_vec_pot = ik_single_batch[1].staggered
        ik_vel_tiled = math.tile(ik_single_batch_vel, [math.shape(self.initial_density)[0]]+[1]*(len(initial_sum.shape)-1))
        ik_vec_pot_tiled = math.tile(ik_single_batch_vec_pot, [math.shape(self.initial_density)[0]]+[1]*(len(initial_sum.shape)-1))
        self.ik_out_velocity = StaggeredGrid(ik_vel_tiled)
        self.ik_out_vec_pot = StaggeredGrid(ik_vec_pot_tiled)
        info("Building force graph...")
        self.force_out, self.forcenet_path = forcenet(self.initial_density, self.initial_velocity, self.target_velocity)

        self.sm_out_by_n = {}
        n = 2
        while n <= len(true_densities):
            self.info("Building SM%d graph..."%n)
            self.sm_out_by_n[n] = GraphBuilder.run_sm(self, n, self.initial_density, self.target_density)
            n *= 2
        self.feed = {}


    def set_dict(self, feed_dict):
        self.feed = feed_dict
