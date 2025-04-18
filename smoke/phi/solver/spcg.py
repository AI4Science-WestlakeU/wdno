from phi.math.nd import *
from phi.solver.base import PressureSolver, conjugate_gradient


class SPCGPressureSolver(PressureSolver):

    def __init__(self):
        PressureSolver.__init__(self, "Single-Phase Conjugate Gradient")

    def solve(self, divergence, active_mask, fluid_mask, boundaries, accuracy, pressure_guess=None,
              max_iterations=500, return_loop_counter=False, gradient_accuracy=None):
        if fluid_mask is not None:
            fluid_mask = boundaries.pad_fluid(fluid_mask)
        # if active_mask is not None:
        #     active_mask = boundaries.pad_active(active_mask)

        def presure_gradient(op, grad):
            return solve_pressure_forward(grad, fluid_mask, max_gradient_iterations, None, gradient_accuracy, boundaries)[0]

        pressure_with_gradient, iteration_count = math.with_custom_gradient(solve_pressure_forward,
                                  [divergence, fluid_mask, max_iterations, pressure_guess, accuracy, boundaries],
                                  presure_gradient,
                                  input_index=0, output_index=0,
                                  name_base="spcg_pressure_solve")

        max_gradient_iterations = max_iterations if gradient_accuracy is not None else iteration_count

        if return_loop_counter:
            return pressure_with_gradient, iteration_count
        else:
            return pressure_with_gradient


def solve_pressure_forward(divergence, fluid_mask, max_iterations, guess, accuracy, boundaries):
    apply_A = lambda pressure: laplace(boundaries.pad_pressure(pressure), weights=fluid_mask, padding="valid")
    return conjugate_gradient(divergence, apply_A, guess, accuracy, max_iterations)
