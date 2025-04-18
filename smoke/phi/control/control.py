from phi.experimental import *
import collections
import random
import math as pymath


class State:
    def __init__(self, frame, density, velocity):
        self.frame = frame
        if isinstance(density, tf.Tensor):
            self.density = density
        else:
            self.density = tf.constant(density)
        if isinstance(velocity, tf.Tensor):
            self.velocity = StaggeredGrid(velocity)
        elif isinstance(velocity, StaggeredGrid):
            self.velocity = velocity
        else:
            self.velocity = StaggeredGrid(tf.constant(velocity))


def simulation_step(sim, state, initial_density):
    velocity = state.velocity
    density = state.density

    # Buoyancy
    velocity += sim.buoyancy(density)

    # Solve pressure
    velocity = sim.divergence_free(velocity)

    # Advect
    density = velocity.advect(density)
    # density = sim.conserve_mass(density, initial_density)
    velocity = velocity.advect(velocity)

    return State(state.frame + 1, density, velocity)


def apply_optimization(state, keyframes, level_scalings, generator_name):
    next_keyframe = [keyframe for keyframe in keyframes if keyframe.frame > state.frame][0]
    remaining_time = next_keyframe.frame - state.frame
    remaining_time = np.array([remaining_time]*state.density.shape[0])
    velocity = state.velocity.staggered

    generator = eval(generator_name)

    added_velocity, (level_scalings, v_by_lvl) = solve_control(state.density, velocity, next_keyframe.density, remaining_time, level_control=level_scalings, velocity_generator=generator)
    velocity += added_velocity

    return State(state.frame, state.density, velocity), (added_velocity, level_scalings, v_by_lvl)



def generate_blocks(shape, blocks=1, blockwidth=3, blockheight=3, total_amount=100, margin=10):
    field = np.zeros(shape, np.float32)
    bs = shape[0]
    m = margin
    for i in range(blocks):
        y = np.random.randint(m,shape[1]-m-1,bs)
        x = np.random.randint(m,shape[2]-m-1,bs)
        for b in range(bs):
            field[b, slice(y[b]-blockheight, y[b]+blockheight), slice(x[b]-blockwidth, x[b]+blockwidth), 0] = 1
    field = field / np.sum(field) * total_amount
    return field


def generate_random_blocks(shape, total_amount=100, margin=10):
    field = np.zeros(shape, np.float32)
    bs = shape[0]
    m = margin
    for b in range(bs):
        ln2_count = random.randint(0, 2)
        ln2_width = random.randint(0, 4 - ln2_count)
        blocks = 2 ** ln2_count
        blockwidth = 2 ** ln2_width
        blockheight = 2 ** (4 - ln2_count - ln2_width)
        for i in range(blocks):
            y = random.randint(m,shape[1]-m-1)
            x = random.randint(m,shape[2]-m-1)
            field[b, slice(y-blockheight, y+blockheight), slice(x-blockwidth, x+blockwidth), 0] = 1
    field = field / np.sum(field) * total_amount
    return field



def multi_scale_control_cnn_16_2(density, velocity, target, remaining_time, level):
    conv = conv_function("Control", constants_file=None)
    reamining_time_tensor = tf.fill(density.shape, 1.0 / remaining_time)

    # Stack information together to create input
    input = tf.concat([density, velocity, target, reamining_time_tensor], axis=-1)

    # Build network
    n = input
    n = conv(n, filters=16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv1")
    n = conv(n, filters=2, kernel_size=[1, 1], padding="same", activation=None,
                                     name="convout", kernel_initializer=tf.initializers.random_normal(0.0, 0.005))
    return n


def small_control_net1(density, velocity, target, remaining_time, level):
    conv = conv_function("Control", constants_file=None)

    reamining_time_tensor = tf.stack([tf.fill(density.shape[1:], 1.0 / remaining_time[i]) for i in range(remaining_time.shape[0])])
    level_tensor = tf.fill(density.shape, 1.0 / (level+1))

    # Stack information together to create input
    input = tf.concat([density, velocity, target, reamining_time_tensor, level_tensor], axis=-1)

    # Build network
    n = input
    n = conv(n, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name="conv1")
    n = conv(n, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="conv2")
    n = conv(n, filters=2, kernel_size=[1, 1], padding="same", activation=None,
                                     name="convout", kernel_initializer=tf.initializers.random_normal(0.0, 0.005))
    return n


def small_level_nets1(density, velocity, target, remaining_time, level):
    conv = conv_function("Control", constants_file=None)

    reamining_time_tensor = tf.stack([tf.fill(density.shape[1:], float(1.0 / remaining_time[i])) for i in range(remaining_time.shape[0])])

    # Stack information together to create input
    input = tf.concat([density, velocity, target, reamining_time_tensor], axis=-1)

    # Build network
    n = input
    n = conv(n, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name="lvl%d_conv1"%level)
    n = conv(n, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="lvl%d_conv2"%level)
    n = conv(n, filters=2, kernel_size=[1, 1], padding="same", activation=None,
                                     name="lvl%d_convout"%level, kernel_initializer=tf.initializers.random_normal(0.0, 0.005))
    return n


def seqnet1(density, velocity, target, remaining_time, level):
    conv = conv_function("Control", constants_file=None)

    reamining_time_tensor = tf.stack([tf.fill(density.shape[1:], float(1.0 / remaining_time[i])) for i in range(remaining_time.shape[0])])

    # Stack information together to create input
    input = tf.concat([density, velocity, target, reamining_time_tensor], axis=-1)

    # Build network
    n = input

    if level == 1:
        n = conv(n, filters=8, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="lvl%d_conv2"%level)
    if level == 2:
        n = conv(n, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name="lvl%d_conv1" % level)
    if level == 3:
        n = conv(n, filters=8, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name="lvl%d_conv1" % level)
        n = conv(n, filters=4, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="lvl%d_conv2"%level)

    if level == 5:
        n = conv(n, filters=8, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name="lvl%d_conv1" % level)
        n = conv(n, filters=4, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name="lvl%d_conv2" % level)

    n = conv(n, filters=2, kernel_size=[1, 1], padding="same", activation=None,
                                     name="lvl%d_convout"%level, kernel_initializer=tf.initializers.random_normal(0.0, 0.005))
    return n


def velocity_variable(density, velocity, target, remaining_time, level):
    return tf.Variable(tf.zeros_like(velocity))


def solve_control(density, velocity, target, remaining_time, level_control=False, velocity_generator=multi_scale_control_cnn_16_2):
    rank = spatial_rank(density)
    size = int(max(density.shape[1:]))

    # Create density grids
    density_grids = [to_dipole_format(density)]
    target_grids = [to_dipole_format(target)]
    velocity_grids = [ velocity ]
    for i in range(pymath.frexp(float(size))[1] - 1): # downsample until 1x1
        density_grids.insert(0, moment_downsample2x(density_grids[0], sum=True))
        target_grids.insert(0, moment_downsample2x(target_grids[0], sum=True))
        velocity_grids.insert(0, downsample2x(velocity_grids[0]))

    added_velocity = None

    level_scalings = []
    velocities_by_level = []

    for i in range(len(density_grids)):
        density_grid = density_grids[i]
        target_grid = target_grids[i]
        velocity_grid = velocity_grids[i]

        if added_velocity is None:
            combined_velocity = velocity_grid
        else:
            added_velocity = upsample2x(added_velocity)[:,2:-2,2:-2,:]
            combined_velocity = velocity_grid + added_velocity

        # Add border pixel for smooth upsampling at the edges
        padded_density = tf.pad(density_grid, [[0,0]]+[[1,1]]*rank+[[0,0]])
        padded_target = tf.pad(target_grid, [[0,0]]+[[1,1]]*rank+[[0,0]])
        padded_velocity = tf.pad(combined_velocity, [[0,0]]+[[1,1]]*rank+[[0,0]])
        if added_velocity is not None:
            added_velocity = tf.pad(added_velocity, [[0,0]]+[[1,1]]*rank+[[0,0]], mode="SYMMETRIC")

        added_velocity_lvl = velocity_generator(padded_density, padded_velocity, padded_target, remaining_time, i)
        velocities_by_level.append(added_velocity_lvl)

        level_scaling = create_level_scaling(level_control, i)
        level_scalings.append(level_scaling)

        if added_velocity is None:
            added_velocity = added_velocity_lvl * level_scaling
        else:
            added_velocity += added_velocity_lvl * level_scaling

    return added_velocity[:,1:-1,1:-1,:], (level_scalings, velocities_by_level)


def neighbour_blur(tensor):
    filter = np.zeros([3, 3, 1, 1])
    filter[1, 1, 0, 0] = 1
    filter[(0, 1, 1, 2), (1, 0, 2, 1), 0, 0] = 1. / 2
    filter[(0, 0, 2, 2), (0, 2, 0, 2), 0, 0] = 1. / 2.41
    filter /= np.sum(filter)
    return tf.nn.conv2d(tensor, filter, [1,1,1,1], "SAME")


def multiscale_blur(density, level_control=False):
    rank = spatial_rank(density)
    size = int(max(density.shape[1:]))

    # Create density grids
    density_grids = [density]
    for i in range(pymath.frexp(float(size))[1] - 2): # downsample until 2x2
        density_grids.insert(0, 2**rank * downsample2x(density_grids[0]))

    potential = None
    level_scalings = []

    for i in range(len(density_grids)):
        blurred_difference = neighbour_blur(density_grids[i])

        level_scaling = create_level_scaling(level_control, i)
        level_scalings.append(level_scaling)

        if potential is None:
            potential = blurred_difference * level_scaling
        else:
            potential = upsample2x(potential)
            potential += blurred_difference * level_scaling

    return potential, (level_scalings, )


def create_level_scaling(level_control, i):
    if isinstance(level_control, float):
        return level_control ** i
    if isinstance(level_control, collections.Iterable):
        return level_control[i]
    elif level_control is True:
        return tf.placeholder(tf.float32, shape=[1, 1, 1, 1], name="lvl_scale_%d" % i)
    elif level_control is False:
        return 1
    else:
        raise ValueError("illegal level_control: {}".format(level_control))


def blur(density, radius, cutoff=None, kernel="1/1+x"):
    with tf.variable_scope("blur"):
        if cutoff is None:
            cutoff = min(int(round(radius * 3)), *density.shape[1:-1])

        xyz = np.meshgrid(*[range(-int(cutoff), (cutoff)+1) for dim in density.shape[1:-1]])
        d = np.sqrt(np.sum([x ** 2 for x in xyz], axis=0))
        if kernel == "1/1+x":
            weights = np.float32(1) / ( d / radius + 1)
        elif kernel.lower() == "gauss":
            weights = math.exp(- d / radius / 2)
        else:
            raise ValueError("Unknown kernel: %s"%kernel)
        weights /= math.sum(weights)
        weights = math.reshape(weights, list(weights.shape) + [1, 1])
        return math.conv(density, weights)


def generate_problem(scenetype, n_frames, sim, problem_count=None):
    bs = problem_count if problem_count is not None else sim.default_batch_size
    if scenetype == "blocksplit":
        initial_density = generate_blocks(sim.shape(batch_size=bs), 1, 4, 4, margin=15)
        initial_velocity = sim.zeros("staggered", batch_size=bs).staggered
        final_density = generate_random_blocks(sim.shape(batch_size=bs))
    elif scenetype == "alphabet_soup":
        from phi.control.voxelutil import alphabet_soup
        initial_density = alphabet_soup(sim.shape(batch_size=bs), random.randint(1, 5), margin=10)
        initial_velocity = sim.zeros("staggered", batch_size=bs).staggered
        final_density = alphabet_soup(sim.shape(batch_size=bs), random.randint(2, 5), margin=12)
    elif scenetype == "test":
        initial_density = sim.zeros(batch_size=bs)
        initial_density[:, 23:31, 9:17,0] = 1
        initial_velocity = sim.zeros("staggered", batch_size=bs).staggered
        final_density = sim.zeros(batch_size=bs)
        final_density[:, 29:33, 33:41,0] = 1
        final_density[:, 12:16, 45:53,0] = 1
    elif scenetype == "word":
        from phi.control.voxelutil import random_word
        initial_density = sim.zeros(batch_size=bs)
        w = initial_density.shape[2]
        initial_density[:, 6:16, w//2-6:w//2+6,0] = 1 # 6:16, 30:40
        initial_density *= 100 / np.sum(initial_density)
        initial_velocity = sim.zeros("staggered", batch_size=bs).staggered
        final_density = random_word(sim.shape(batch_size=bs), 2, 5, margin=12, total_content=100, y=30)
    else:
        raise ValueError("Unknown scene type: %s"%scenetype)

    if problem_count is None:
        initial_state = State(0, initial_density, initial_velocity)
        final_state = State(n_frames, final_density, sim.zeros("staggered"))
    else:
        initial_state = State(0, sim.placeholder(), sim.placeholder("staggered"))
        final_state = State(n_frames, sim.placeholder(), sim.placeholder("staggered"))

    return initial_state, final_state, initial_density, initial_velocity, final_density