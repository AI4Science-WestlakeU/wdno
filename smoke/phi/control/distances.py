from phi.math.nd import *


def _initial_dijkstra_tensor(target_mask):
    finite_mask = target_mask != 0
    array = np.ones_like(target_mask, np.float32) * np.inf
    array[finite_mask] = 0
    return array


def _dijkstra_step(tensor): # (batch, spatial_dims, 1)
    rank = spatial_rank(tensor)
    dims = range(rank)
    all_dims = range(len(tensor.shape))
    center_min = tensor
    for dimension in dims:
        padded = np.pad(tensor, [[1,1] if i-1 == dimension else [0,0] for i in all_dims], "constant", constant_values=np.inf)
        upper_slices = [(slice(2, None) if i-1 == dimension else slice(None)) for i in all_dims]
        lower_slices = [(slice(-2) if i-1 == dimension else slice(None)) for i in all_dims]
        center_min_dim = np.minimum(padded[lower_slices]+1, padded[upper_slices]+1)
        center_min = np.minimum(center_min, center_min_dim)
    return center_min


def l1_distance_map(target_mask, fluid_mask=None, non_fluid_value=-1):
    """
Calculates the shortest distance from all grid points to the nearest entry in target_mask.
Neighbouring cells are separated by distance 1. All resulting distances are integers.
    :param target_mask: Mask encoding points of distance 0. Shape (batch, spatial_dims..., 1)
    :param fluid_mask: Mask encoding the domain topology (same shape as target_mask)
    :param non_fluid_value: This value will be used in the returned tensor for non-fluid cells (fluid_mask==0).
    :return: A tensor of same shape as target_mask containing the shortest distances
    """
    tensor = _initial_dijkstra_tensor(target_mask)
    obstacle_cell_count = 0 if fluid_mask is None else np.sum(fluid_mask == 0)
    while True:
        prev_tensor = tensor
        tensor = _dijkstra_step(tensor)
        if fluid_mask is not None:
            tensor[fluid_mask == 0] = np.inf

        if len(tensor[~np.isfinite(tensor)]) == obstacle_cell_count:
            tensor[~np.isfinite(tensor)] = non_fluid_value
            return tensor
        if np.array_equal(prev_tensor, tensor):
            raise ValueError("Unconnected regions detected, failed to create distance map")


def shortest_halfway_point(distribution1, distribution2, fluid_mask=None):
    distances1 = l1_distance_map(distribution1, fluid_mask, non_fluid_value=np.inf)
    distances2 = l1_distance_map(distribution2, fluid_mask, non_fluid_value=np.inf)
    with np.errstate(invalid="ignore"):
        equal_points = np.abs(distances1-distances2) <= 1
    # Find lowest indices
    min_dist = np.min(distances1[equal_points])
    shortest_equal_points = equal_points & (distances1 == min_dist)
    indices = np.argwhere(shortest_equal_points)
    mean_index = np.mean(indices, 0)
    return np.round(mean_index).astype(np.int)


# fluid_mask = np.ones([1,8,8,1])
# fluid_mask[:, 4, 1:7, :] = 0
# p1 = np.zeros([1,8,8,1])
# p1[:,0, 4, :] = 1
# p2 = np.zeros([1,8,8,1])
# p2[:, 7, 4, :] = 1
# print(shortest_halfway_point(p1, p2, fluid_mask))