from phi.math.base import DynamicBackend
backend = DynamicBackend()

from phi.math.scipy_backend import SciPyBackend
backend.backends.append(SciPyBackend())


def load_tensorflow():
    """
Internal function to register the TensorFlow backend.
This function is called automatically once a TFSimulation is instantiated.
    :return: True if TensorFlow could be imported, else False
    """
    try:
        import phi.math.tensorflow_backend as tfb
        for b in backend.backends:
            if isinstance(b, tfb.TFBackend): return True
        backend.backends.append(tfb.TFBackend())
        return True
    except BaseException as e:
        import logging
        logging.fatal("Failed to load TensorFlow backend. Error: %s" % e)
        print("Failed to load TensorFlow backend. Error: %s" % e)
        return False


abs = backend.abs
add = backend.add
boolean_mask = backend.boolean_mask
ceil = backend.ceil
floor = backend.floor
concat = backend.concat
conv = backend.conv
dimrange = backend.dimrange
dot = backend.dot
exp = backend.exp
expand_dims = backend.expand_dims
flatten = backend.flatten
gather = backend.gather
isfinite = backend.isfinite
matmul = backend.matmul
max = backend.max
maximum = backend.maximum
mean = backend.mean
minimum = backend.minimum
name = backend.name
ones_like = backend.ones_like
pad = backend.pad
py_func = backend.py_func
resample = backend.resample
reshape = backend.reshape
shape = backend.shape
sqrt = backend.sqrt
stack = backend.stack
std = backend.std
sum = backend.sum
tile = backend.tile
to_float = backend.to_float
unstack = backend.unstack
while_loop = backend.while_loop
with_custom_gradient = backend.with_custom_gradient
zeros_like = backend.zeros_like