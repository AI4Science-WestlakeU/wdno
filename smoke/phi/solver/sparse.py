import scipy, scipy.sparse, scipy.sparse.linalg
from phi.math.nd import *
from phi.solver.base import ExplicitBoundaryPressureSolver, conjugate_gradient


class SparseSciPyPressureSolver(ExplicitBoundaryPressureSolver):

    def __init__(self):
        super(SparseSciPyPressureSolver, self).__init__("Sparse SciPy")

    def solve_with_boundaries(self, divergence, active_mask, fluid_mask, accuracy, pressure_guess=None, **kwargs):
        dimensions = list(divergence.shape[1:-1])
        A = sparse_pressure_matrix(dimensions, active_mask, fluid_mask)

        def np_solve_p(div):
            div_vec = div.reshape([-1, A.shape[0]])
            pressure = [scipy.sparse.linalg.spsolve(A, div_vec[i, ...]) for i in range(div_vec.shape[0])]
            return np.array(pressure).reshape(div.shape).astype(np.float32)

        def np_solve_p_gradient(op, grad_in):
            return math.py_func(np_solve_p, [grad_in], np.float32)

        pressure = math.py_func(np_solve_p, [divergence], np.float32, divergence.shape, grad=np_solve_p_gradient)
        return pressure


def sparse_pressure_matrix(dimensions, extended_active_mask, extended_fluid_mask):
    """
Builds a sparse matrix such that when applied to a flattened pressure field, it calculates the laplace
of that field, taking into account obstacles and empty cells.
    :param dimensions: valid simulation dimensions. Pressure field should be of shape (batch size, dimensions..., 1)
    :param extended_active_mask: Binary tensor with 2 more entries in every dimension than "dimensions".
    :param extended_fluid_mask: Binary tensor with 2 more entries in every dimension than "dimensions".
    :return: SciPy sparse matrix that acts as a laplace on a flattened pressure field given obstacles and empty cells
    """
    N = int(np.prod(dimensions))
    d = len(dimensions)
    A = scipy.sparse.lil_matrix((N, N), dtype=np.float32)
    dims = range(d)

    center_values = None # diagonal matrix entries

    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions)) # d * (N^2) array mapping from linear to spatial indices

    for dim in dims:
        upper_indices = [slice(None)] + [slice(2, None) if i == dim else slice(1, -1) for i in dims] + [slice(None)]
        center_indices = [slice(None)] + [slice(1, -1) if i == dim else slice(1, -1) for i in dims] + [slice(None)]
        lower_indices = [slice(None)] + [slice(0, -2) if i == dim else slice(1, -1) for i in dims] + [slice(None)]

        self_active = extended_active_mask[center_indices]
        stencil_upper = extended_active_mask[upper_indices] * self_active
        stencil_lower = extended_active_mask[lower_indices] * self_active
        stencil_center = - extended_fluid_mask[upper_indices] - extended_fluid_mask[lower_indices]

        if center_values is None:
            center_values = math.flatten(stencil_center)
        else:
            center_values = center_values + math.flatten(stencil_center)

        # Find entries in matrix

        dim_direction = np.zeros_like(gridpoints)
        dim_direction[dim] = 1
        # Upper indices
        upper_indices = gridpoints + dim_direction
        upper_in_range_inx = np.nonzero(upper_indices[dim] < dimensions[dim])
        upper_indices_linear = np.ravel_multi_index(upper_indices[:,upper_in_range_inx], dimensions)
        A[gridpoints_linear[upper_in_range_inx], upper_indices_linear] = stencil_upper.flatten()[upper_in_range_inx]
        # Lower indices
        lower_indices = gridpoints - dim_direction
        lower_in_range_inx = np.nonzero(lower_indices[dim] >= 0)
        lower_indices_linear = np.ravel_multi_index(lower_indices[:, lower_in_range_inx], dimensions)
        A[gridpoints_linear[lower_in_range_inx], lower_indices_linear] = stencil_lower.flatten()[lower_in_range_inx]

    A[gridpoints_linear, gridpoints_linear] = math.minimum(center_values, -1)

    return scipy.sparse.csc_matrix(A)



class SparseCGPressureSolver(ExplicitBoundaryPressureSolver):

    def __init__(self, autodiff=False):
        super(SparseCGPressureSolver, self).__init__("Sparse Conjugate Gradient")
        self.autodiff = autodiff

    def solve_with_boundaries(self, divergence, active_mask, fluid_mask, accuracy, pressure_guess=None,
                              max_iterations=500, gradient_accuracy=None, return_loop_counter=False):
        dimensions = list(divergence.shape[1:-1])
        N = int(np.prod(dimensions))

        if math.backend.choose_backend(divergence).matches_name("TensorFlow"):
            import tensorflow as tf
            sidx, sorting = sparse_indices(dimensions)
            sval_data = sparse_values(dimensions, active_mask, fluid_mask, sorting)
            # coo = A.tocoo()
            # indices = np.mat([coo.row, coo.col]).transpose()
            A = tf.SparseTensor(indices=sidx, values=sval_data, dense_shape=[N, N])
        else:
            A = sparse_pressure_matrix(dimensions, active_mask, fluid_mask)

        if self.autodiff:
            pressure, iter = sparse_cg(divergence, A, max_iterations, pressure_guess, accuracy, back_prop=True)
        else:
            def pressure_gradient(op, grad):
                return sparse_cg(grad, A, max_gradient_iterations, None, gradient_accuracy)[0]

            pressure, iter = math.with_custom_gradient(sparse_cg,
                                                       [divergence, A, max_iterations, pressure_guess, accuracy],
                                                       pressure_gradient, input_index=0, output_index=0,
                                                       name_base="scg_pressure_solve")

        max_gradient_iterations = max_iterations if gradient_accuracy is not None else iter

        if return_loop_counter:
            return pressure, iter
        else:
            return pressure


def sparse_cg(divergence, A, max_iterations, guess, accuracy, back_prop=False):
    div_vec = math.reshape(divergence, [-1, int(np.prod(divergence.shape[1:]))])
    if guess is not None:
        guess = math.reshape(guess, [-1, int(np.prod(divergence.shape[1:]))])
    apply_A = lambda pressure: math.matmul(A, pressure)
    result_vec, iterations = conjugate_gradient(div_vec, apply_A, guess, accuracy, max_iterations, back_prop)
    return math.reshape(result_vec, math.shape(divergence)), iterations



def sparse_indices(dimensions):
    N = int(np.prod(dimensions))
    d = len(dimensions)
    dims = range(d)

    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions)) # d * (N^2) array mapping from linear to spatial indices

    indices_list = [ np.stack([gridpoints_linear] * 2, axis=-1) ]

    for dim in dims:
        dim_direction = np.zeros_like(gridpoints)
        dim_direction[dim] = 1
        # Upper indices
        upper_indices = gridpoints + dim_direction
        upper_in_range_inx = np.nonzero(upper_indices[dim] < dimensions[dim])
        upper_indices_linear = np.ravel_multi_index(upper_indices[:,upper_in_range_inx], dimensions)[0,:]
        indices_list.append(np.stack([gridpoints_linear[upper_in_range_inx], upper_indices_linear], axis=-1))
        # Lower indices
        lower_indices = gridpoints - dim_direction
        lower_in_range_inx = np.nonzero(lower_indices[dim] >= 0)
        lower_indices_linear = np.ravel_multi_index(lower_indices[:, lower_in_range_inx], dimensions)[0,:]
        indices_list.append(np.stack([gridpoints_linear[lower_in_range_inx], lower_indices_linear], axis=-1))

    indices = np.concatenate(indices_list, axis=0)

    sorting = np.lexsort(np.transpose(indices)[:,::-1])

    sorted_indices = indices[sorting]

    return sorted_indices, sorting


def sparse_values(dimensions, extended_active_mask, extended_fluid_mask, sorting=None):
    """
Builds a sparse matrix such that when applied to a flattened pressure field, it calculates the laplace
of that field, taking into account obstacles and empty cells.
    :param dimensions: valid simulation dimensions. Pressure field should be of shape (batch size, dimensions..., 1)
    :param extended_active_mask: Binary tensor with 2 more entries in every dimension than "dimensions".
    :param extended_fluid_mask: Binary tensor with 2 more entries in every dimension than "dimensions".
    :return: SciPy sparse matrix that acts as a laplace on a flattened pressure field given obstacles and empty cells
    """
    N = int(np.prod(dimensions))
    d = len(dimensions)
    dims = range(d)

    values_list = []
    center_values = None # diagonal matrix entries

    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions)) # d * (N^2) array mapping from linear to spatial indices

    for dim in dims:
        upper_indices = [slice(None)] + [slice(2, None) if i == dim else slice(1, -1) for i in dims] + [slice(None)]
        center_indices = [slice(None)] + [slice(1, -1) if i == dim else slice(1, -1) for i in dims] + [slice(None)]
        lower_indices = [slice(None)] + [slice(0, -2) if i == dim else slice(1, -1) for i in dims] + [slice(None)]

        self_active = extended_active_mask[center_indices]
        stencil_upper = extended_active_mask[upper_indices] * self_active
        stencil_lower = extended_active_mask[lower_indices] * self_active
        stencil_center = - extended_fluid_mask[upper_indices] - extended_fluid_mask[lower_indices]

        if center_values is None:
            center_values = math.flatten(stencil_center)
        else:
            center_values = center_values + math.flatten(stencil_center)

        dim_direction = np.zeros_like(gridpoints)
        dim_direction[dim] = 1
        # Upper indices
        upper_indices = gridpoints + dim_direction
        upper_in_range_inx = np.nonzero(upper_indices[dim] < dimensions[dim])[0]
        values_list.append(math.gather(math.flatten(stencil_upper), upper_in_range_inx))
        # Lower indices
        lower_indices = gridpoints - dim_direction
        lower_in_range_inx = np.nonzero(lower_indices[dim] >= 0)[0]
        values_list.append(math.gather(math.flatten(stencil_lower), lower_in_range_inx))

    center_values = math.minimum(center_values, -1.)
    values_list.insert(0, center_values)

    values = math.concat(values_list, axis=0)
    if sorting is not None:
        values = math.gather(values, sorting)
    return values