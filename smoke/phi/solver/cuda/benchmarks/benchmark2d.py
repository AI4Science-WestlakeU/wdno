from matplotlib.ticker import FormatStrFormatter

from phi.backend.base import load_tensorflow
from phi.solver.cuda.cuda import CudaPressureSolver
from phi.solver.sparse import SparseCGPressureSolver
import matplotlib.pyplot as plt
from phi.solver.cuda.benchmarks.benchmark_utils import *

cudaSolver = CudaPressureSolver()
sparseCGSolver = SparseCGPressureSolver()

# configuration of the benchmark
warmup = 5
testruns = 25

dimension = 2
accuracy = 1e-5
batch_size = 1

cpuTests = [] #[16, 32, 64]#, 128, 256, 512]#, 1024]#, 2048]
tfTests = [] #[16, 32, 64, 128, 256, 512, 1024]#, 2048]
cudaTests = [16, 32, 64, 128, 256, 512, 1024, 2048]

# benchmark
load_tensorflow()
cudaTimes = benchmark_pressure_solve(cudaSolver, cudaTests, dimension, tf.float32, warmup, testruns, accuracy, batch_size)
tfTimes = benchmark_pressure_solve(sparseCGSolver, tfTests, dimension, tf.float64, warmup, testruns, accuracy, batch_size)
cpuTimes = benchmark_pressure_solve(sparseCGSolver, cpuTests, dimension, tf.float64, warmup, testruns, accuracy, batch_size, cpu=True)

cudaAVG = [np.mean(a) for a in cudaTimes]
cudaSTD = [np.std(a) for a in cudaTimes]
tfAVG = [np.mean(a) for a in tfTimes]
tfSTD = [np.std(a) for a in tfTimes]
cpuAVG = [np.mean(a) for a in cpuTimes]
cpuSTD = [np.std(a) for a in cpuTimes]

# serialize and print all data necessary for the graph
print("cudaTests = " + str(cudaTests))
print("cudaAVG = " + str(cudaAVG))
print("cudaSTD = " + str(cudaSTD))
print("tfTests = " + str(tfTests))
print("tfAVG = " + str(tfAVG))
print("tfSTD = " + str(tfSTD))
print("cpuTests = " + str(cpuTests))
print("cpuAVG = " + str(cpuAVG))
print("cpuSTD = " + str(cpuSTD))

plt.errorbar(tfTests, tfAVG, tfSTD, fmt='-o')
plt.errorbar(cpuTests, cpuAVG, cpuSTD, fmt='-o')
plt.errorbar(cudaTests, cudaAVG, cudaSTD, fmt='-o')

plt.legend(['Tensorflow GPU', 'Tensorflow CPU', 'CUDA'], loc='upper left')
plt.xscale('log', basex=2)
plt.yscale('log')
plt.xticks(cudaTests)
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.xlabel("Grid Dimension 2D")
plt.ylabel("Computation Time in seconds")
plt.show()
