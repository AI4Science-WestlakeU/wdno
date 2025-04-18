from phi.math.nd import *
import tensorflow as tf

def conv_pressure(divergence):
    comps = np.meshgrid(*[range(-dim, dim+1) for dim in divergence.shape[1:-1]])
    d = np.sqrt(np.sum([comp**2 for comp in comps], axis=0))
    weights = - np.float32(1) / np.maximum(d, 0.5) # / (4*np.pi)
    weights = np.reshape(weights, list(d.shape)+[1, 1])
    return tf.nn.conv2d(divergence, weights, [1, 1, 1, 1], "SAME")