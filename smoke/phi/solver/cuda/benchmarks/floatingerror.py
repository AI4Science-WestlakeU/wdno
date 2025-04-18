from matplotlib.ticker import FormatStrFormatter

from phi.solver.cuda.cuda import CudaPressureSolver
from phi.solver.sparse import SparseCGPressureSolver, load_tensorflow
import matplotlib.pyplot as plt
from phi.solver.cuda.benchmarks.benchmark_utils import *

cudaSolver = CudaPressureSolver()
numpySolver = SparseCGPressureSolver()
load_tensorflow()

testruns = 20
accuracy = 1e-5
tests2d = [16, 32, 64, 128, 256, 512, 1024]
dimension = 2

error2dAbs, error2dRel = benchmark_error(cudaSolver, numpySolver, tests2d, dimension, testruns, accuracy)

error2dAbsAVG = [np.mean(a) for a in error2dAbs]
error2dAbsSTD = [np.std(a) for a in error2dAbs]
error2dRelAVG = [np.mean(a) for a in error2dRel]
error2dRelSTD = [np.std(a) for a in error2dRel]

print('tests2d = ' + str(tests2d))
print('error2dAbsAVG = ' + str(error2dAbsAVG))
print('error2dAbsSTD = ' + str(error2dAbsSTD))
print('error2dRelAVG = ' + str(error2dRelAVG))
print('error2dRelSTD = ' + str(error2dRelSTD))

gc.collect()

tests3d = [16, 32, 64, 128]
dimension = 3

error3dAbs, error3dRel = benchmark_error(cudaSolver, numpySolver, tests3d, dimension, testruns, accuracy)

error3dAbsAVG = [np.mean(a) for a in error3dAbs]
error3dAbsSTD = [np.std(a) for a in error3dAbs]
error3dRelAVG = [np.mean(a) for a in error3dRel]
error3dRelSTD = [np.std(a) for a in error3dRel]


print('tests2d = ' + str(tests2d))
print('error2dAbsAVG = ' + str(error2dAbsAVG))
print('error2dAbsSTD = ' + str(error2dAbsSTD))
print('error2dRelAVG = ' + str(error2dRelAVG))
print('error2dRelSTD = ' + str(error2dRelSTD))

print('tests3d = ' + str(tests3d))
print('error3dAbsAVG = ' + str(error3dAbsAVG))
print('error3dAbsSTD = ' + str(error3dAbsSTD))
print('error3dRelAVG = ' + str(error3dRelAVG))
print('error3dRelSTD = ' + str(error3dRelSTD))


plt.errorbar(tests2d, error2dAbsAVG, error2dAbsSTD, fmt='-o')
plt.errorbar(tests2d, error2dRelAVG, error2dRelSTD, fmt='-o')
plt.errorbar(tests3d, error3dAbsAVG, error3dAbsSTD, fmt='-o')
plt.errorbar(tests3d, error3dRelAVG, error3dRelSTD, fmt='-o')

plt.legend(['Absolute error 2D', 'Relative error 2D', 'Absolute error 3D', 'Relative error 3D'], loc='bottom right')
plt.xscale('log', basex=2)
plt.yscale('log')
plt.xticks(tests2d)
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.xlabel("Grid Dimension")
plt.ylabel("Error compared to CPU result")
plt.show()










