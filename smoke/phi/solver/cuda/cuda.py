from phi.solver.base import ExplicitBoundaryPressureSolver
import tensorflow as tf
from phi import math
import numpy as np


class CudaPressureSolver(ExplicitBoundaryPressureSolver):

    def __init__(self):
        super(CudaPressureSolver, self).__init__("CUDA Conjugate Gradient")
        import os
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.pressure_op = tf.load_op_library(current_dir + "/build/pressure_solve_op.so")

    def solve_with_boundaries(self, divergence, active_mask, fluid_mask, accuracy=1e-5, pressure_guess=None,  # pressure_guess is not used in this implementation => Kernel automatically takes the last pressure value for initial_guess
                              max_iterations=2000, gradient_accuracy=None, return_loop_counter=False):

        def pressure_gradient(op, grad):
            return self.cuda_solve_forward(grad, active_mask, fluid_mask, accuracy, max_iterations)[0]

        pressure_out, iterations = math.with_custom_gradient(self.cuda_solve_forward,
                                                             [divergence, active_mask, fluid_mask, accuracy, max_iterations],
                                                             pressure_gradient, input_index=0, output_index=0, name_base="cuda_pressure_solve")

        if return_loop_counter:
            return pressure_out, iterations
        else:
            return pressure_out

    def cuda_solve_forward(self, divergence, active_mask, fluid_mask, accuracy, max_iterations):
        dimensions = divergence.get_shape()[1:-1]
        dimensions = dimensions[::-1]  # the custom op needs it in the x,y,z order
        dim_array = np.array(dimensions)
        dim_product = np.prod(dimensions)

        mask_dimensions = dim_array + 2

        laplace_matrix = tf.zeros(dim_product * (len(dimensions) * 2 + 1), dtype=tf.int8)

        # Helper variables for CG, make sure new memory is allocated for each variable.
        one_vector = tf.ones(dim_product, dtype=tf.float32)
        p = tf.zeros_like(divergence, dtype=tf.float32) + 1
        z = tf.zeros_like(divergence, dtype=tf.float32) + 2
        r = tf.zeros_like(divergence, dtype=tf.float32) + 3
        pressure = tf.zeros_like(divergence, dtype=tf.float32) + 4

        # Call the custom kernel
        pressure_out, iterations = self.pressure_op.pressure_solve(dimensions,

                                                                   mask_dimensions,
                                                                   active_mask,
                                                                   fluid_mask,
                                                                   laplace_matrix,

                                                                   divergence,
                                                                   p, r, z, pressure, one_vector,

                                                                   dim_product,
                                                                   accuracy,
                                                                   max_iterations)
        return pressure_out, iterations