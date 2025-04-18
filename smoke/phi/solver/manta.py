import numpy as np
import tensorflow as tf
import mantatensor.mantatensor_bindings as mt
import mantatensor.mantatensor_gradients # registers gradients
from phi.solver.base import PressureSolver


class MantaSolver(PressureSolver):

    def __init__(self):
        super(MantaSolver, self).__init__("Manta")

    def solve(self, divergence, accuracy=1e-05):
        return mt_solve_pressure(divergence, self.fluid_mask, accuracy)

    def set_fluid_mask(self, fluid_mask):
        self.fluid_mask = fluid_mask



def mt_solve_pressure(divergence, fluid_mask, accuracy):
    dimensions = list(divergence.shape[1:-1])

    neg_div = - divergence
    batches = neg_div.shape[0]
    try:
        batches = int(batches)
    except:
        raise ValueError("Manta solver requires fixed batch size")

    if len(dimensions) == 3:
        velocity_mac = tf.zeros([batches] + dimensions + [3])
        scalar_shape_mt = neg_div.shape
    else:
        scalar_shape_mt = [batches, 1] + dimensions + [1]
        velocity_mac = tf.zeros([batches, 1] + dimensions + [3])
        neg_div = tf.reshape(neg_div, scalar_shape_mt)

    flags_tensor = tf.constant(flags_array(fluid_mask, dimensions), name='flag_grid')
    flags_tensor = tf.tile(flags_tensor, [batches, 1, 1, 1, 1])


    pressure_var = tf.Variable(np.zeros(scalar_shape_mt, dtype=np.float32), name='pressure')
    pressure_out = mt.solve_pressure_system(neg_div, velocity_mac, pressure_var, flags_tensor, 1, batches,
                                            cgAccuracy=accuracy)

    return to_tensorflow_scalar(pressure_out, dimensions)


def flags_array(fluid_mask, dimensions):
    flags = (2 - fluid_mask).astype(np.int32)
    if len(dimensions) == 3:
        return flags.reshape((1,) + flags.shape)
    elif len(dimensions) == 2:
        return flags.reshape((1, 1) + flags.shape)
    else:
        raise ValueError("Only 2 and 3 dimensions supported")


def to_tensorflow_scalar(field, dimensions):
    if len(field.shape) != len(dimensions) + 2:
        field = field[:, 0, :, :, :]
    return field


def to_tensorflow_vector(field, dimensions):
    if len(field.shape) != len(dimensions) + 2:
        field = field[:, 0, :, :, :]
    if field.shape[-1] != len(dimensions):
        field = field[..., 0:len(dimensions)]
    return field


# def _to_mantaflow_3vec(field):
#     if isinstance(field, StaggeredGrid):
#         field = field.staggered
#     if field.shape[-1] == 2:
#         backend.pad(field, [[0,0]]*(len(field.shape)-1) + [[0,1]])
#     if len(field.shape) == 4:
#         field = backend.reshape(field, [1] + list(field.shape))
#
#     if len(field.shape) != 5 or field.shape[-1] != 3:
#         raise ValueError("Cannot convert field of shape {} to mantaflow".format(field.shape))
#     return field