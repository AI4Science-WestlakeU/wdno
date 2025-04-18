import collections, math, os.path, inspect
from phi.solver.base import PressureSolver
from phi.experimental import *


class NetworkSolver(PressureSolver):

    def __init__(self):
        super(NetworkSolver, self).__init__("Net")

    def solve(self, divergence, active_mask, fluid_mask, boundaries, accuracy, pressure_guess=None, **kwargs):
        base_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
        ckpt = os.path.join(base_path, "data/pnet_/modelsave.ckpt")
        return solve_pressure_tompson2(divergence, level_control=False, constants_file=ckpt, **kwargs)[0]




def tompson2_pressure(div, constants_file=None):
    conv = conv_function("Tompson2", constants_file=constants_file)
    n = div
    n = conv(n, filters=8, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv1")
    n = conv(n, filters=16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv2")
    n = conv(n, filters=1, kernel_size=[1, 1], padding="same", activation=None, name="conv_out")
    return n


def tompson2_load(sess, graph, path="./data/tompson2/modelsave.ckpt"):
    restore_net("Tompson2", sess, graph, path)


def solve_pressure_tompson2(divergence, level_control=False, constants_file=None, cubic=True):
    rank = spatial_rank(divergence)
    dV = 2**rank
    size = int(max(divergence.shape[1:]))

    # if cubic:
    #     resize = tf.image.resize_bicubic
    # else:
    #     resize = tf.image.resize_bilinear

    multires_div = [to_dipole_format(divergence)] # order from low-res to high-res
    for i in range(math.frexp(float(size))[1] - 2): # downsample until 2x2
        p = downsample_dipole_2d_2x(multires_div[0], scaling="sum")
        # p = tf.layers.average_pooling2d(multires_div[0], pool_size=[2, 2], strides=2) * dV
        multires_div.insert(0, p)

    p_div = None # Divergence of pressure
    pressure = None

    pressure_accum = []
    pressure_by_lvl = []
    p_div_accum = []
    p_div_by_level = []
    remaining_div = []
    level_scalings = []
    i = 0

    for div_lvl in multires_div: # start with low-res
        div = div_lvl

        if p_div is not None: # Upsample previous level and subtract div p
            double_shape = np.array(pressure.shape[1:-1]) * 2
            p_div = upsample2x(p_div) / dV
            pressure = upsample2x(pressure)[:,2:-2,2:-2,:]
            # p_div = resize(p_div, div.shape[1:-1]) / dV
            # pressure = resize(pressure, double_shape)[:,2:-2,2:-2,:]
            div -= to_dipole_format(p_div)

        normalized_div, std = normalize_dipole(div)
        padded_div = tf.pad(normalized_div, [[0,0]]+[[1,1]]*rank+[[0,0]])
        if pressure is not None: pressure = tf.pad(pressure, [[0,0]]+[[1,1]]*rank+[[0,0]], mode="SYMMETRIC")
        pressure_lvl = std * tompson2_pressure(padded_div, constants_file=constants_file)
        delta_p_div = laplace(pressure_lvl)[:, 1:-1, 1:-1, :]

        pressure_by_lvl.append(pressure_lvl)
        p_div_by_level.append(delta_p_div)

        if isinstance(level_control, collections.Iterable):
            level_scaling = level_control[i]
        elif level_control is True:
            level_scaling = tf.placeholder(tf.float32, shape=[1, 1, 1, 1], name="lvl_scale_%d"%i)
        elif level_control is False:
            level_scaling = 1
        else:
            raise ValueError("illegal level_control: {}".format(level_control))
        level_scalings.append(level_scaling)

        if p_div is None:
            pressure = pressure_lvl * level_scaling
            p_div = delta_p_div * level_scaling
        else:
            pressure += pressure_lvl * level_scaling
            p_div += delta_p_div * level_scaling

        pressure_accum.append(pressure)
        i += 1

    return pressure[:,1:-1,1:-1,:], (p_div, pressure_accum, level_scalings)
