import numpy as np
import phi.math as math


def spatial_rank(tensor_or_mac):
    """
Returns the number of spatial dimensions.
Arrays are expected to be of the shape (batch size, spatial dimensions..., component size)
The number of spatial dimensions is equal to the tensor rank minus two.
    :param tensor_or_mac: a tensor or StaggeredGrid instance
    :return: the number of spatial dimensions as an integer
    """
    if isinstance(tensor_or_mac, StaggeredGrid):
        return tensor_or_mac.spatial_rank
    else:
        return len(tensor_or_mac.shape) - 2


def indices_tensor(tensor, dtype=np.float32):
    """
Returns an index tensor of the same spatial shape as the given tensor.
Each index denotes the location within the tensor starting from zero.
Indices are encoded as vectors in the index tensor.
    :param tensor: a tensor of shape (batch size, spatial dimensions..., component size)
    :param dtype: a numpy data type (default float32)
    :return: an index tensor of shape (1, spatial dimensions..., spatial rank)
    """
    spatial_dimensions = list(tensor.shape[1:-1])
    idx_zyx = np.meshgrid(*[range(dim) for dim in spatial_dimensions], indexing="ij")
    idx = np.stack(idx_zyx, axis=-1).reshape([1, ] + spatial_dimensions + [len(spatial_dimensions)])
    return idx.astype(dtype)


def normalize_to(target, source=1):
    """
Multiplies the target so that its total content matches the source.
    :param target: a tensor
    :param source: a tensor or number
    :return: normalized tensor of the same shape as target
    """
    return target * (math.sum(source) / math.sum(target))


def l1_loss(tensor, batch_norm=True, reduce_batches=True):
    if isinstance(tensor, StaggeredGrid):
        tensor = tensor.staggered
    if reduce_batches:
        total_loss = math.sum(math.abs(tensor))
    else:
        total_loss = math.sum(math.abs(tensor), axis=list(range(1, len(tensor.shape))))
    if batch_norm and reduce_batches:
        batch_size = math.shape(tensor)[0]
        return total_loss / math.to_float(batch_size)
    else:
        return total_loss


def l2_loss(tensor, batch_norm=True, reduce_batches=True):
    return l_n_loss(tensor, 2, batch_norm=batch_norm, reduce_batches=reduce_batches)


def l_n_loss(tensor, n, batch_norm=True, reduce_batches=True):
    if isinstance(tensor, StaggeredGrid):
        tensor = tensor.staggered
    if reduce_batches:
        total_loss = math.sum(tensor ** n) / n
    else:
        total_loss = math.sum(tensor ** n, axis=list(range(1, len(tensor.shape)))) / n

    if batch_norm:
        batch_size = math.shape(tensor)[0]
        return total_loss / math.to_float(batch_size)
    else:
        return total_loss


def at_centers(field):
    if isinstance(field, StaggeredGrid):
        return field.at_centers()
    else:
        return field


# Divergence

def divergence(vel, dx=1, difference="central"):
    """
Computes the spatial divergence of a vector field from finite differences.
    :param vel: tensor of shape (batch size, spatial dimensions..., spatial rank) or StaggeredGrid
    :param dx: distance between adjacent grid points (default 1)
    :param difference: type of difference, one of ('forward', 'central') (default 'forward')
    :return: tensor of shape (batch size, spatial dimensions..., 1)
    """
    if isinstance(vel, StaggeredGrid):
        return vel.divergence()

    assert difference in ('central', 'forward')
    rank = spatial_rank(vel)
    if difference == "forward":
        return _forward_divergence_nd(vel) / dx ** rank
    else:
        return _central_divergence_nd(vel) / (2 * dx) ** rank


def _forward_divergence_nd(field):
    rank = spatial_rank(field)
    dims = range(rank)
    components = []
    for dimension in dims:
        vq = field[...,rank-dimension-1]
        upper_slices = [(slice(1, None) if i == dimension else slice(None)) for i in dims]
        lower_slices = [(slice(-1)      if i == dimension else slice(None)) for i in dims]
        diff = vq[[slice(None)]+upper_slices] - vq[[slice(None)]+lower_slices]
        padded = math.pad(diff, [[0,0]] + [([0,1] if i==dimension else [0,0]) for i in dims])
        components.append(padded)
    return math.expand_dims(math.add(components), -1)


def _central_divergence_nd(tensor):
    rank = spatial_rank(tensor)
    dims = range(rank)
    components = []
    tensor = math.pad(tensor, [[0, 0]] + [[1, 1]]*rank + [[0, 0]])
    for dimension in dims:
        upper_slices = [(slice(2, None) if i == dimension else slice(1, -1)) for i in dims]
        lower_slices = [(slice(-2) if i == dimension else slice(1, -1)) for i in dims]
        diff = tensor[[slice(None)] + upper_slices + [rank - dimension - 1]] - \
               tensor[[slice(None)] + lower_slices + [rank - dimension - 1]]
        components.append(diff)
    return math.expand_dims(math.add(components), -1)


# Gradient

def gradient(tensor, dx=1, difference="forward"):
    """
Calculates the gradient of a scalar field from finite differences.
The gradient vectors are in reverse order, lowest dimension first.
    :param tensor: field with shape (batch_size, spatial_dimensions..., 1)
    :param dx: physical distance between grid points (default 1)
    :param difference: type of difference, one of ('forward', 'backward', 'central') (default 'forward')
    :return: tensor of shape (batch_size, spatial_dimensions..., spatial rank)
    """
    if tensor.shape[-1] != 1: raise ValueError("Gradient requires a scalar field as input")
    dims = range(spatial_rank(tensor))
    field = tensor[...,0]

    if 1 in field.shape[1:]:
        raise ValueError("All spatial dimensions must have size larger than 1, got {}".format(tensor.shape))

    if difference.lower() == "central":
        return _central_diff_nd(tensor, dims) / (dx * 2)
    elif difference.lower() == "forward":
        return _forward_diff_nd(field, dims) / dx
    elif difference.lower() == "backward":
        return _backward_diff_nd(field, dims) / dx
    else:
        raise ValueError("Invalid difference type: {}. Can be CENTRAL or FORWARD".format(difference))


def _backward_diff_nd(field, dims):
    df_dq = []
    for dimension in dims:
        upper_slices = [(slice(1, None) if i==dimension else slice(None)) for i in dims]
        lower_slices = [(slice(-1)      if i==dimension else slice(None)) for i in dims]
        diff = field[[slice(None)]+upper_slices] - field[[slice(None)]+lower_slices]
        padded = math.pad(diff, [[0,0]]+[([1,0] if i == dimension else [0,0]) for i in dims])
        df_dq.append(padded)
    return math.stack(df_dq[::-1], axis=-1)


def _forward_diff_nd(field, dims):
    df_dq = []
    for dimension in dims:
        upper_slices = [(slice(1, None) if i==dimension else slice(None)) for i in dims]
        lower_slices = [(slice(-1)      if i==dimension else slice(None)) for i in dims]
        diff = field[[slice(None)]+upper_slices] - field[[slice(None)]+lower_slices]
        padded = math.pad(diff, [[0,0]]+[([0,1] if i == dimension else [0,0]) for i in dims])
        df_dq.append(padded)
    return math.stack(df_dq[::-1], axis=-1)


def _central_diff_nd(field, dims):
    field = math.pad(field, [[0,0]] + [[1,1]]*spatial_rank(field) + [[0, 0]], "symmetric")
    df_dq = []
    for dimension in dims:
        upper_slices = [(slice(2, None) if i==dimension else slice(1,-1)) for i in dims]
        lower_slices = [(slice(-2)      if i==dimension else slice(1,-1)) for i in dims]
        diff = field[[slice(None)] + upper_slices + [0]] - field[[slice(None)] + lower_slices + [0]]
        df_dq.append(diff)
    return math.stack(df_dq[::-1], axis=-1)


# Laplace

def laplace(tensor, weights=None, padding="symmetric"):
    if tensor.shape[-1] != 1:
        raise ValueError("Laplace operator requires a scalar field as input")
    rank = spatial_rank(tensor)

    if padding.lower() != "valid":
        tensor = math.pad(tensor, [[0,0]] + [[1,1]] * rank + [[0,0]], padding)

    if weights is not None:
        return _weighted_sliced_laplace_nd(tensor, weights)

    if rank == 2:
        return _conv_laplace_2d(tensor)
    elif rank == 3:
        return _conv_laplace_3d(tensor)
    else:
        return _sliced_laplace_nd(tensor)


def _conv_laplace_2d(tensor):
    kernel = np.zeros((3, 3, 1, 1), np.float32)
    kernel[1,1,0,0] = -4
    kernel[(0,1,1,2),(1,0,2,1),0,0] = 1
    return math.conv(tensor, kernel, padding="VALID")


def _conv_laplace_3d(tensor):
    kernel = np.zeros((3, 3, 3, 1, 1), np.float32)
    kernel[1,1,1,0,0] = -6
    kernel[(0,1,1,1,1,2), (1,0,2,1,1,1), (1,1,1,0,2,1), 0,0] = 1
    return math.conv(tensor, kernel, padding="VALID")


def _sliced_laplace_nd(tensor):
    # Laplace code for n dimensions
    dims = range(spatial_rank(tensor))
    components = []
    for dimension in dims:
        center_slices = [(slice(1, -1) if i == dimension else slice(1,-1)) for i in dims]
        upper_slices = [(slice(2, None) if i == dimension else slice(1,-1)) for i in dims]
        lower_slices = [(slice(-2) if i == dimension else slice(1,-1)) for i in dims]
        diff = tensor[[slice(None)] + upper_slices + [slice(None)]] \
               + tensor[[slice(None)] + lower_slices + [slice(None)]] \
               - 2 * tensor[[slice(None)] + center_slices + [slice(None)]]
        components.append(diff)
    return math.add(components)


def _weighted_sliced_laplace_nd(tensor, weights):
    if tensor.shape[-1] != 1: raise ValueError("Laplace operator requires a scalar field as input")
    dims = range(spatial_rank(tensor))
    components = []
    for dimension in dims:
        center_slices = [(slice(1, -1) if i == dimension else slice(1,-1)) for i in dims]
        upper_slices = [(slice(2, None) if i == dimension else slice(1,-1)) for i in dims]
        lower_slices = [(slice(-2) if i == dimension else slice(1,-1)) for i in dims]

        lower_weights = weights[[slice(None)] + lower_slices + [slice(None)]] * weights[[slice(None)] + center_slices + [slice(None)]]
        upper_weights = weights[[slice(None)] + upper_slices + [slice(None)]] * weights[[slice(None)] + center_slices + [slice(None)]]
        center_weights = - lower_weights - upper_weights

        lower_values = tensor[[slice(None)] + lower_slices + [slice(None)]]
        upper_values = tensor[[slice(None)] + upper_slices + [slice(None)]]
        center_values = tensor[[slice(None)] + center_slices + [slice(None)]]

        diff = upper_values * upper_weights + lower_values * lower_weights + center_values * center_weights
        components.append(diff)
    return math.add(components)



# Downsample / Upsample

def downsample2x(tensor, interpolation="LINEAR"):
    if interpolation.lower() != "linear":
        raise ValueError("Only linear interpolation supported")
    dims = range(spatial_rank(tensor))
    tensor = math.pad(tensor, [[0,0]]+
                          [([0, 1] if (dim % 2) != 0 else [0,0]) for dim in tensor.shape[1:-1]]
                          + [[0,0]], "SYMMETRIC")
    for dimension in dims:
        upper_slices = [(slice(1, None, 2) if i==dimension else slice(None)) for i in dims]
        lower_slices = [(slice(0, None, 2) if i==dimension else slice(None)) for i in dims]
        sum = tensor[[slice(None)]+upper_slices+[slice(None)]] + tensor[[slice(None)]+lower_slices+[slice(None)]]
        tensor = sum / 2
    return tensor


def upsample2x(tensor, interpolation="LINEAR"):
    if interpolation.lower() != "linear":
        raise ValueError("Only linear interpolation supported")
    dims = range(spatial_rank(tensor))
    vlen = tensor.shape[-1]
    spatial_dims = tensor.shape[1:-1]
    tensor = math.pad(tensor, [[0, 0]] + [[1, 1]]*spatial_rank(tensor) + [[0, 0]], "SYMMETRIC")
    for dim in dims:
        left_slices_1 =  [(slice(2, None) if i==dim else slice(None)) for i in dims]
        left_slices_2 =  [(slice(1,-1)    if i==dim else slice(None)) for i in dims]
        right_slices_1 = [(slice(1, -1)   if i==dim else slice(None)) for i in dims]
        right_slices_2 = [(slice(-2)      if i==dim else slice(None)) for i in dims]
        left = 0.75 * tensor[[slice(None)]+left_slices_2+[slice(None)]] + 0.25 * tensor[[slice(None)]+left_slices_1+[slice(None)]]
        right = 0.25 * tensor[[slice(None)]+right_slices_2+[slice(None)]] + 0.75 * tensor[[slice(None)]+right_slices_1+[slice(None)]]
        combined = math.stack([right, left], axis=2+dim)
        tensor = math.reshape(combined, [-1] + [spatial_dims[dim] * 2 if i == dim else tensor.shape[i+1] for i in dims] + [vlen])
    return tensor


def spatial_sum(tensor):
    if isinstance(tensor, StaggeredGrid):
        tensor = tensor.staggered
    summed = math.sum(tensor, axis=math.dimrange(tensor))
    for i in math.dimrange(tensor):
        summed = math.expand_dims(summed, i)
    return summed


class StaggeredGrid:
    """
        MACGrids represent a staggered vector field in which each vector component is sampled at the
        face centers of centered hypercubes.
        Going in the direction of a vector component, the first entry samples the lower face of the first cube and the
        last entry the upper face of the last cube.
        Therefore staggered grids contain one more entry in each spatial dimension than a centered field.
        This results in oversampling in the other directions. There, highest element lies outside the grid.

        Attributes:
            shape (tuple Tensorshape): the shape of the staggered field
            staggered (tensor): array or tensor holding the staggered field

        """
    def __init__(self, staggered):
        self.staggered = staggered

    def __repr__(self):
        return "StaggeredGrid(shape=%s)" % self.shape

    def at_centers(self):
        rank = self.spatial_rank
        dims = range(rank)
        df_dq = []
        for d in dims:  # z,y,x
            upper_slices = [(slice(1, None) if i == d else slice(-1)) for i in dims]
            lower_slices = [(slice(-1) if i == d else slice(-1)) for i in dims]
            sum = self.staggered[[slice(None)] + upper_slices + [rank - d - 1]] +\
                  self.staggered[[slice(None)] + lower_slices + [rank - d - 1]]
            df_dq.append(sum / rank)
        return math.stack(df_dq[::-1], axis=-1)

    def at_faces(self, face_dimension_xyz):
        dims = range(self.spatial_rank)
        face_dimension_zyx = len(dims) - face_dimension_xyz - 1  # 0=Z, 1=Y, 2=X, etc.
        components = []
        for d in dims:  # z,y,x
            if d == face_dimension_zyx:
                components.append(self.staggered[..., len(dims) - d - 1])
            else:
                # Interpolate other components
                vq = self.staggered[..., len(dims) - d - 1]
                t = vq
                for d2 in dims:  # z,y,x
                    slices1 = [(slice(1, None) if i == d2 else slice(None)) for i in dims]
                    slices2 = [(slice(-1) if i == d2 else slice(None)) for i in dims]
                    t = t[[slice(None)] + slices1] + t[[slice(None)] + slices2]
                    if d2 == d:
                        t = math.pad(t, [[0, 0]] + [([0, 1] if i == d2 else [0, 0]) for i in dims]) / 2
                    else:
                        t = math.pad(t, [[0, 0]] + [([1, 0] if i == d2 else [0, 0]) for i in dims]) / 2
                components.append(t)

        return math.stack(components[::-1], axis=-1)

    def divergence(self):
        dims = range(self.spatial_rank)
        components = []
        for dimension in dims:
            comp = self.spatial_rank - dimension - 1
            upper_slices = [(slice(1, None) if i == dimension else slice(-1)) for i in dims]
            lower_slices = [(slice(-1) if i == dimension else slice(-1)) for i in dims]
            diff = self.staggered[[slice(None)] + upper_slices + [comp]] - \
                   self.staggered[[slice(None)] + lower_slices + [comp]]
            components.append(diff)
        return math.expand_dims(math.add(components), -1)

    def abs(self):
        return StaggeredGrid(math.abs(self.staggered))

    def length_squared(self):
        centered = self.at_centers()
        scalar = math.sum(centered ** 2, axis=-1)
        return math.expand_dims(scalar, axis=-1)

    def soft_sqrt(self):
        return StaggeredGrid(math.sqrt(math.maximum(self.staggered, 1e-20)))

    def normalize(self):
        v_length = math.sqrt(math.add([self.staggered[..., i] ** 2 for i in range(self.shape[-1])]))
        global_mean = math.mean(v_length, axis=range(1, self.spatial_rank+1))
        for i in range(self.spatial_rank+1):
            global_mean = math.expand_dims(global_mean, -1)
        return StaggeredGrid(self.staggered / global_mean)

    def total(self):
        v_length = math.sqrt(math.add([self.staggered[..., i] ** 2 for i in range(self.shape[-1])]))
        total = math.sum(v_length, axis=range(1, self.spatial_rank+1))
        for i in range(self.spatial_rank+1):
            total = math.expand_dims(total, -1)
        return total

    def batch_div(self, tensor):
        return StaggeredGrid(self.staggered / tensor)

    def advect(self, field, interpolation="LINEAR", dt=1):
        """
    Performs a semi-Lagrangian advection step, propagating the field through the velocity field.
    A backwards Euler step is performed and the smpling is performed according to the interpolation specified.
        :param field: scalar or vector field to propagate
        :param velocity: vector field specifying the velocity at each point in space. Shape (batch_size, grid_size,
        :param dt:
        :param interpolation: LINEAR, BSPLINE, IDW (default is LINEAR)
        :return: the advected field
        """
        if isinstance(field, StaggeredGrid):
            return self.multi_advect([field], interpolation=interpolation, dt=dt)[0]
        else:
            return self._advect_centered_field(field, dt, interpolation)

    def _advect_centered_field(self, field, dt, interpolation):
        idx = indices_tensor(field)
        centered_velocity = self.at_centers()[..., ::-1]  # assume right number of components
        sample_coords = idx - centered_velocity * dt
        result = math.resample(field, sample_coords, interpolation=interpolation, boundary="REPLICATE")
        return result

    def _advect_mac(self, field_mac, dt, interpolation):
        # resample in each dimension
        idx = indices_tensor(self.staggered)
        advected_component_fields = []
        dims = range(len(self.staggered.shape) - 2)

        for d in dims:  # z,y,x
            velocity_at_staggered_points = self.at_faces(len(dims) - d - 1)[..., ::-1]
            sample_coords = idx - velocity_at_staggered_points * dt
            d_comp = len(dims) - d - 1
            advected = math.resample(field_mac[..., d_comp:d_comp + 1], sample_coords, interpolation=interpolation,
                                   boundary="REPLICATE")
            advected_component_fields.append(advected)

        all_advected = math.concat(advected_component_fields[::-1], axis=-1)
        return all_advected

    def multi_advect(self, fields, interpolation="LINEAR", dt=1):
        assert isinstance(fields, (list, tuple)), "first parameter must be either a tuple or list"
        inputs_lists = []
        coords_lists = []
        value_generators = []
        for field in fields:
            if isinstance(field, StaggeredGrid):
                i, c, v = self._mac_block_advection(field.staggered, dt)
            else:
                i, c, v = self._centered_block_advection(field, dt)
            inputs_lists.append(i)
            coords_lists.append(c)
            value_generators.append(v)

        inputs = math.concat(sum(inputs_lists, []), 0)
        coords = math.concat(sum(coords_lists, []), 0)
        all_advected = math.resample(inputs, coords, interpolation=interpolation, boundary="REPLICATE")
        all_advected = math.reshape(all_advected, [self.spatial_rank, -1] + list(all_advected.shape[1:]))
        all_advected = math.unstack(all_advected)
        results = []
        abs_i = 0
        for i in range(len(inputs_lists)):
            n = len(inputs_lists[0])
            assigned_advected = all_advected[abs_i:abs_i+n]
            results.append(value_generators[i](assigned_advected))
            abs_i += n
        return results

    def _mac_block_advection(self, field_mac, dt):
        # resample in each dimension
        idx = indices_tensor(self.staggered)
        dims = range(len(self.staggered.shape) - 2)
        inputs_list = []
        coords_list = []

        for d in dims:  # z,y,x
            velocity_at_staggered_points = self.at_faces(len(dims) - d - 1)[..., ::-1]
            sample_coords = idx - velocity_at_staggered_points * dt
            d_comp = len(dims) - d - 1
            coords_list.append(sample_coords)
            inputs_list.append(field_mac[..., d_comp:d_comp + 1])

        def post_advection(advected_list):
            return StaggeredGrid(math.concat(advected_list[::-1], axis=-1))

        return inputs_list, coords_list, post_advection

    def _centered_block_advection(self, field, dt):
        idx = indices_tensor(field)
        centered_velocity = self.at_centers()[..., ::-1]  # assume right number of components
        sample_coords = idx - centered_velocity * dt
        return [field], [sample_coords], lambda list: list[0]

    def curl(self):
        rank = spatial_rank(self.staggered)
        if rank == 3:
            return self._staggered_curl_3d()
        elif rank == 2:
            return self._staggered_curl_2d()
        else:
            raise ValueError("Curl requires a two or three-dimensional vector field")

    def pad(self, lower, upper=None, mode="symmetric"):
        upper = upper if upper is not None else lower
        padded = math.pad(self.staggered, [[0,0]] + [[lower,upper]]*self.spatial_rank + [[0,0]], mode)
        return StaggeredGrid(padded)

    def _staggered_curl_3d(self):
        """
    Calculates the curl operator on a staggered three-dimensional field.
    The resulting vector field is a staggered grid.
    If the velocities of the vector potential were sampled at the lower faces of a cube, the resulting velocities
    are sampled at the centers of the upper edges.
        :param vector_potential: three-dimensional vector potential
        :return: three-dimensional staggered vector field
        """
        kernel = np.zeros((2, 2, 2, 3, 3), np.float32)
        derivative = np.array([-1, 1])
        # x-component: dz/dy - dy/dz
        kernel[0, :, 0, 2, 0] = derivative
        kernel[:, 0, 0, 1, 0] = -derivative
        # y-component: dx/dz - dz/dx
        kernel[:, 0, 0, 0, 1] = derivative
        kernel[0, 0, :, 2, 1] = -derivative
        # z-component: dy/dx - dx/dy
        kernel[0, 0, :, 1, 2] = derivative
        kernel[0, :, 0, 0, 2] = -derivative

        vector_potential = math.pad(self.staggered, [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0]], "SYMMETRIC")
        vector_field = math.conv(vector_potential, kernel, padding="VALID")
        return StaggeredGrid(vector_field)

    def _staggered_curl_2d(self):
        kernel = np.zeros((2, 2, 1, 2), np.float32)
        derivative = np.array([-1, 1])
        # x-component: dz/dy
        kernel[:, 0, 0, 0] = derivative
        # y-component: - dz/dx
        kernel[0, :, 0, 1] = -derivative

        scalar_potential = math.pad(self.staggered, [[0, 0], [0, 1], [0, 1], [0, 0]], "SYMMETRIC")
        vector_field = math.conv(scalar_potential, kernel, padding="VALID")
        return StaggeredGrid(vector_field)

    def __add__(self, other):
        if isinstance(other, StaggeredGrid):
            return StaggeredGrid(self.staggered + other.staggered)
        else:
            return StaggeredGrid(self.staggered + other)

    def __sub__(self, other):
        if isinstance(other, StaggeredGrid):
            return StaggeredGrid(self.staggered - other.staggered)
        else:
            return StaggeredGrid(self.staggered - other)

    def __mul__(self, other):
        if isinstance(other, StaggeredGrid):
            return StaggeredGrid(self.staggered * other.staggered)
        else:
            return StaggeredGrid(self.staggered * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, StaggeredGrid):
            return StaggeredGrid(self.staggered / other.staggered)
        else:
            return StaggeredGrid(self.staggered / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __getitem__(self, item):
        return StaggeredGrid(self.staggered[item])

    @property
    def shape(self):
        return self.staggered.shape

    @property
    def tensor_rank(self):
        return len(self.staggered.shape)

    @property
    def spatial_rank(self):
        return spatial_rank(self.staggered)

    @property
    def name(self):
        try:
            return self.staggered.name
        except:
            return None

    @staticmethod
    def gradient(scalar_field, padding="symmetric"):
        if scalar_field.shape[-1] != 1: raise ValueError("Gradient requires a scalar field as input")
        rank = spatial_rank(scalar_field)
        dims = range(rank)
        field = math.pad(scalar_field, [[0,0]]+[[1,1]]*rank+[[0,0]], mode=padding)
        df_dq = []
        for dimension in dims:
            upper_slices = [(slice(1, None) if i==dimension else slice(1, None)) for i in dims]
            lower_slices = [(slice(-1)      if i==dimension else slice(1, None)) for i in dims]
            diff = field[[slice(None)]+upper_slices] - field[[slice(None)]+lower_slices]
            df_dq.append(diff)
        return StaggeredGrid(math.concat(df_dq[::-1], axis=-1))

    @staticmethod
    def from_scalar(scalar_field, axis_forces, padding_mode="constant"):
        if scalar_field.shape[-1] != 1: raise ValueError("Resample requires a scalar field as input")
        rank = spatial_rank(scalar_field)
        dims = range(rank)
        df_dq = []
        for dimension in dims: # z,y,x
            padded_field = math.pad(scalar_field, [[0,0]]+[[1,1] if i == dimension else [0,1] for i in dims]+[[0,0]], padding_mode)
            upper_slices = [(slice(1, None) if i == dimension else slice(None)) for i in dims]
            lower_slices = [(slice(-1) if i == dimension else slice(None)) for i in dims]
            neighbour_sum = padded_field[[slice(None)] + upper_slices + [slice(None)]] + \
                            padded_field[[slice(None)] + lower_slices + [slice(None)]]
            df_dq.append(axis_forces[dimension] * neighbour_sum * 0.5 / rank)
        return StaggeredGrid(math.concat(df_dq[::-1], axis=-1))