import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from phi.solver.cuda.benchmarks.benchmark_utils import *

testruns = 25
warmup = 5
tests = [8, 16, 32, 64, 128]#, 256, 512, 1024, 2048]
dimension = 3

cudaResults = benchmark_laplace_matrix_cuda(tests, dimension, warmup, testruns)
gc.collect()
phiResults = benchmark_laplace_matrix_phi(tests, dimension, warmup, testruns)

cudaAVG = [np.mean(a) for a in cudaResults]
cudaSTD = [np.std(a) for a in cudaResults]
phiAVG = [np.mean(a) for a in phiResults]
phiSTD = [np.std(a) for a in phiResults]

print("tests = " + str(tests))
print("cudaAVG = " + str(cudaAVG))
print("cudaSTD = " + str(cudaSTD))
print("phiAVG = " + str(phiAVG))
print("phiSTD = " + str(phiSTD))



plt.errorbar(tests, cudaAVG, cudaSTD, fmt='-o')
plt.errorbar(tests, phiAVG, phiSTD, fmt='-o')


plt.legend(['Cuda', 'PhiFlow'], loc='upper left')
plt.xscale('log', basex=2)
plt.yscale('log')
plt.xticks(tests)
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.xlabel("Grid Dimension")
plt.ylabel("Computation Time in seconds")
plt.show()
