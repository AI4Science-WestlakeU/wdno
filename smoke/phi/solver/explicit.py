from phi.experimental import *



def explicit_dipole_pressure(div, num=1):
    # [filter_height, filter_width, in_channels, out_channels]
    filter = np.zeros([3, 3, 3, 3], np.float32)
    # pressure (q)
    filter[(0, 1, 1, 2), (1, 0, 2, 1), 0, 0] = 1 # edges q
    filter[(0, 1, 1, 2), (1, 0, 2, 1), (2, 1, 1, 2), 0] = (+0.0986, +0.0986, -0.0986, -0.0986) # edges px, py
    filter[(0, 0, 2, 2), (0, 2, 0, 2), 0, 0] = 0.7071 # corners q
    filter[(0, 0, 2, 2), (0, 2, 0, 2), 1, 0] = (0.03288, -0.03288, -0.03288, 0.03288) # corners px
    filter[(0, 0, 2, 2), (0, 2, 0, 2), 2, 0] = (0.03288, 0.03288, -0.03288, -0.03288) # corners py
    filter[1, 1, 0, 0] = 1.4142 # self-pressure
    # pressure gradient
    filter[(0, 1, 1, 2), (1, 0, 2, 1), 0, (2, 1, 1, 2)] = (-0.5, -0.5, +0.5, +0.5) # edges q
    filter[(0, 1, 1, 2), (1, 0, 2, 1), (2, 1, 1, 2), (2, 1, 1, 2)] = (-0.2347, -0.2347, 0.2347, 0.2347) # edges px, py longitudinal
    filter[(0, 1, 1, 2), (1, 0, 2, 1), (1, 2, 2, 1), (1, 2, 2, 1)] = (0.2347/4, 0.2347/4, 0.2347/4, 0.2347/4) # edges px, py transversal
    filter[(0, 0, 2, 2), (0, 2, 0, 2), 0, 1] = (-0.3536, +0.3536, -0.3536, +0.3536) # corners q -> px
    filter[(0, 0, 2, 2), (0, 2, 0, 2), 0, 2] = (-0.3536, -0.3536, +0.3536, +0.3536) # corners q -> py
    filter[1, 1, (1, 2), (1, 2)] = 1./num # self-pressure
    # corners px,py -> px,py is comparably small
    return tf.nn.conv2d(div, filter, strides=[1, 1, 1, 1], padding="SAME")


def explicit_pressure_multigrid(divergence, level_control=False):
    rank = spatial_rank(divergence)
    dV = 2**rank
    size = int(max(divergence.shape[1:]))

    multires_div = [to_dipole_format(divergence)] # order from low-res to high-res
    for i in range(math.frexp(float(size))[1] - 2): # downsample until 2x2
        p = downsample_dipole_2d_2x(multires_div[0])
        # p = tf.layers.average_pooling2d(multires_div[0], pool_size=[2, 2], strides=2) * dV
        multires_div.insert(0, p)

    p_div = None # Divergence of pressure
    pressure = None

    pressure_accum = []
    pressure_by_lvl = []
    p_div_by_level = []
    level_scalings = []
    i = 0

    for div_lvl in multires_div: # start with low-res
        div = div_lvl

        if p_div is not None: # Upsample previous level and subtract div p
            div -= to_dipole_format(p_div)

        pressure_lvl = explicit_dipole_pressure(div, num=len(multires_div))
        pressure_lvl = upsample_flatten_dipole_2d_2x(pressure_lvl)
        delta_p_div = laplace(pressure_lvl)

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
            pressure = upsample2x(pressure) + pressure_lvl * level_scaling
            p_div = upsample2x(p_div) / dV + delta_p_div * level_scaling

        pressure_accum.append(pressure)
        i += 1

    pressure = tf.layers.average_pooling2d(pressure, [2, 2], [2, 2])
    p_div = tf.layers.average_pooling2d(p_div, [2, 2], [2, 2])
    return pressure, (p_div, pressure_accum, level_scalings)
