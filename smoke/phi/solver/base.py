# coding=utf-8
from phi.math.nd import *


def create_mask(divergence_tensor):
    return np.ones([1]+list(divergence_tensor.shape)[1:], np.float32)


class PressureSolver(object):

    def __init__(self, name):
        self.name = name

    def solve(self, divergence, active_mask, fluid_mask, boundaries, accuracy, pressure_guess=None, **kwargs):
        """
Solves the pressure equation Δp = ∇·v for all active fluid cells where active cells are given by the active_mask.
The resulting pressure is expected to fulfill (Δp-∇·v) ≤ accuracy for every active cell.
        :param divergence: the scalar divergence of the velocity field, ∇·v
        :param active_mask: (Optional) Scalar field encoding active cells as ones and inactive (open/obstacle) as zero.
        :param fluid_mask: (Optional) Scalar field encoding fluid cells as ones and obstacles as zero.
        Has the same dimensions as the divergence field. If no obstacles are present, None may be passed.
        :param boundaries: DomainBoundary object defining open and closed boundaries
        :param accuracy: The accuracy of the result. Every grid cell should fulfill (Δp-∇·v) ≤ accuracy
        :param pressure_guess: (Optional) Pressure field which can be used as an initial state for the solver
        :param kwargs: solver-specific arguments
        """
        raise NotImplementedError()


class ExplicitBoundaryPressureSolver(PressureSolver):

    def __init__(self, name):
        PressureSolver.__init__(self, name)

    def solve(self, divergence, active_mask, fluid_mask, boundaries, accuracy, pressure_guess=None, **kwargs):
        active_mask = create_mask(divergence) if active_mask is None else active_mask
        active_mask = boundaries.pad_active(active_mask)
        fluid_mask = create_mask(divergence) if fluid_mask is None else fluid_mask
        fluid_mask = boundaries.pad_fluid(fluid_mask)
        return self.solve_with_boundaries(divergence, active_mask, fluid_mask, accuracy, pressure_guess, **kwargs)

    def solve_with_boundaries(self, divergence, active_mask, fluid_mask, accuracy, pressure_guess=None, **kwargs):
        """
See :func:`PressureSolver.solve`. Unlike the regular solve method, active_mask and fluid_mask are valid tensors which include
one extra voxel at each boundary to account for boundary conditions.
        :param divergence: n^d dimensional scalar field
        :param active_mask: (n+2)^d dimensional scalar field
        :param fluid_mask: (n+2)^d dimensional scalar field
        :param accuracy:
        :param pressure_guess:
        :param kwargs:
        """
        raise NotImplementedError()


def conjugate_gradient(k, apply_A, initial_x=None, accuracy=1e-5, max_iterations=1024, back_prop=False):
    """
Solve the linear system of equations Ax=k using the conjugate gradient (CG) algorithm.
The implementation is based on https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf
    :param k: Right-hand-side vector
    :param apply_A: function that takes x and calculates Ax
    :param initial_x: initial guess for the value of x
    :param accuracy: the algorithm terminates once |Ax-k| ≤ accuracy for every element. If None, the algorithm runs until max_iterations is reached.
    :param max_iterations: maximum number of CG iterations to perform
    :return: Pair containing the result for x and the number of iterations performed
    """
    if initial_x is None:
        x = math.zeros_like(k)
        momentum = k
    else:
        x = initial_x
        momentum = k - apply_A(x)
    residual = momentum

    laplace_momentum = apply_A(momentum)
    loop_index = 0

    vars = [x, momentum, laplace_momentum, residual, loop_index]

    if accuracy is not None:
        def loop_condition(_1, _2, _3, residual, i):
            return math.max(math.abs(residual)) >= accuracy
    else:
        def loop_condition(_1, _2, _3, residual, i):
            return True

    def loop_body(pressure, momentum, A_times_momentum, residual, loop_index):
        tmp = math.sum(momentum * A_times_momentum)
        a = math.sum(momentum * residual) / tmp
        pressure += a * momentum
        residual -= a * A_times_momentum
        b = - math.sum(residual * A_times_momentum) / tmp
        momentum = residual + b * momentum
        A_times_momentum = apply_A(momentum)
        return [pressure, momentum, A_times_momentum, residual, loop_index + 1]

    x, momentum, laplace_momentum, residual, loop_index = math.while_loop(loop_condition, loop_body, vars,
                                                                              parallel_iterations=2, back_prop=back_prop,
                                                                              swap_memory=False,
                                                                              name="pressure_solve_loop",
                                                                              maximum_iterations=max_iterations)

    return x, loop_index
