import time
import sys
import numpy as np
import tensorflow as tf
import gc
from phi.solver.sparse import sparse_indices, sparse_values


# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write(('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)))
    # Print New Line on Complete
    if iteration == total:
        print()


def benchmark_pressure_solve_numpy(solver, tests, dimension, warmup=10, testruns=50, accuracy=1e-5, batch_size=1):
    result = []
    for n in tests:

        active_mask = np.ones([1] + [n] * dimension + [1])
        active_mask = np.pad(active_mask, [[0, 0]] + [[1, 1]] * dimension + [[0, 0]], mode='constant')

        # open boundary
        fluid_mask = np.ones([1] + [n + 2] * dimension + [1])

        # closed boundary => with random divergence to expensive to solve
        # fluid_mask = np.ones([1] + [n] * dimension + [1])
        # fluid_mask = np.pad(fluid_mask, [[0, 0]] + [[1, 1]] * dimension + [[0, 0]], mode='constant')

        testresults = []

        # Warmup
        print(n)
        print("warmup...")
        print_progress_bar(0, warmup)
        for i in range(warmup):
            randomDivergence = np.random.uniform(low=-1, high=1, size=([batch_size] + [n] * dimension + [1]))
            solver.solve_with_boundaries(randomDivergence, active_mask, fluid_mask, accuracy=accuracy,
                                         max_iterations=10000)
            print_progress_bar(i + 1, warmup)

        print_progress_bar(0, testruns)
        for i in range(testruns):
            randomDivergence = np.random.uniform(low=-1, high=1, size=([batch_size] + [n] * dimension + [1]))

            start = time.time()
            solver.solve_with_boundaries(randomDivergence, active_mask, fluid_mask, accuracy=accuracy,
                                         max_iterations=10000)
            end = time.time()

            testresults.append(end - start)
            print_progress_bar(i + 1, testruns)

        testresults = np.array(testresults)
        result.append(testresults)
        print(n)
        print(np.mean(testresults))
        print(np.std(testresults))
        print()

    return result


def benchmark_pressure_solve(solver, tests, dimension, dtype, warmup=10, testruns=50, accuracy=1e-5, batch_size=1,
                             cpu=False):
    result = []
    for n in tests:

        active_mask = np.ones([1] + [n] * dimension + [1])
        active_mask = np.pad(active_mask, [[0, 0]] + [[1, 1]] * dimension + [[0, 0]], mode='constant')

        # open boundary
        fluid_mask = np.ones([1] + [n + 2] * dimension + [1])

        # closed boundary => with random divergence to expensive to solve
        # fluid_mask = np.ones([1] + [n] * dimension + [1])
        # fluid_mask = np.pad(fluid_mask, [[0, 0]] + [[1, 1]] * dimension + [[0, 0]], mode='constant')

        testresults = []
        div = tf.get_variable("divergence",
                              initializer=tf.constant(np.zeros(([batch_size] + [n] * dimension + [1])), dtype=dtype))

        if cpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto()

        with tf.Session(config=config):
            pressure = solver.solve_with_boundaries(div, active_mask, fluid_mask, accuracy=accuracy,
                                                    max_iterations=10000)
            tf.global_variables_initializer().run()

            # Warmup
            print(n)
            print("warmup...")
            print_progress_bar(0, warmup)
            for i in range(warmup):
                randomDivergence = np.random.uniform(low=-1, high=1, size=([batch_size] + [n] * dimension + [1]))
                div.assign(randomDivergence).eval()
                pressure.eval()
                print_progress_bar(i + 1, warmup)

            print_progress_bar(0, testruns)
            for i in range(testruns):
                randomDivergence = np.random.uniform(low=-1, high=1, size=([batch_size] + [n] * dimension + [1]))
                div.assign(randomDivergence).eval()

                start = time.time()
                pressure.eval()
                end = time.time()

                testresults.append(end - start)
                print_progress_bar(i + 1, testruns)

        testresults = np.array(testresults)
        result.append(testresults)
        print(n)
        print(np.mean(testresults))
        print(np.std(testresults))
        print()
        tf.reset_default_graph()

    return result


def benchmark_error(cudaSolver, numpySolver, tests, dimension, testruns=50, accuracy=1e-5):
    errorAbsResult = []
    errorRelResult = []
    for n in tests:
        active_mask = np.ones([1] + [n] * dimension + [1])
        active_mask = np.pad(active_mask, [[0, 0]] + [[1, 1]] * dimension + [[0, 0]], mode='constant')

        # open boundary
        fluid_mask = np.ones([1] + [n + 2] * dimension + [1])

        # closed boundary => with random divergence to expensive to solve
        # fluid_mask = np.ones([1] + [n] * dimension + [1])
        # fluid_mask = np.pad(fluid_mask, [[0, 0]] + [[1, 1]] * dimension + [[0, 0]], mode='constant')

        testresultsAbs = []
        testresultsRel = []

        with tf.Session(""):
            div = tf.get_variable("divergence",
                                  initializer=tf.constant(np.zeros(([1] + [n] * dimension + [1])), dtype=tf.float32))
            div2 = tf.get_variable("divergence2",
                                   initializer=tf.constant(np.zeros(([1] + [n] * dimension + [1])), dtype=tf.float64))
            pressureKernel = cudaSolver.solve_with_boundaries(div, active_mask, fluid_mask, accuracy=accuracy,
                                                              max_iterations=10000)
            tf.global_variables_initializer().run()

            print_progress_bar(0, testruns)
            for i in range(testruns):
                randomDivergence = np.random.uniform(low=-1, high=1, size=([1] + [n] * dimension + [1]))
                div.assign(randomDivergence).eval()
                div2.assign(randomDivergence).eval()

                ref = numpySolver.solve_with_boundaries(div2, active_mask, fluid_mask, accuracy=accuracy,
                                                        max_iterations=10000).eval().flatten()
                pressure = pressureKernel.eval().flatten()

                for j in range(len(pressure)):
                    errorAbs = abs(pressure[j] - ref[j])
                    if ref[j] != 0:
                        errorRel = abs(errorAbs / ref[j])
                    else:
                        errorRel = abs(pressure[j])

                    testresultsAbs.append(errorAbs)
                    testresultsRel.append(errorRel)

                gc.collect()

                print_progress_bar(i + 1, testruns)

        testresultsAbs = np.array(testresultsAbs)
        testresultsRel = np.array(testresultsRel)
        errorAbsResult.append(testresultsAbs)
        errorRelResult.append(testresultsRel)
        print(n)
        print(np.mean(testresultsAbs))
        print(np.mean(testresultsRel))
        print()
        tf.reset_default_graph()
        gc.collect()

    return errorAbsResult, errorRelResult


def benchmark_laplace_matrix_cuda(tests, dimension, warmup=10, testruns=50):
    import os
    current_dir = os.path.dirname(os.path.realpath(__file__))
    laplace_op = tf.load_op_library(current_dir + "/../build/laplace_op.so")

    resultCuda = []

    for n in tests:
        dimensions = ([n] * dimension)
        mask_dimensions = ([n + 2] * dimension)
        dim_product = n ** dimension
        active_mask = np.ones([1] + [n] * dimension + [1])
        active_mask = np.pad(active_mask, [[0, 0]] + [[1, 1]] * dimension + [[0, 0]], mode='constant')
        fluid_mask = np.ones([1] + [n + 2] * dimension + [1])

        testresultsCuda = []

        with tf.Session(""):
            zeros = np.zeros(dim_product * (dimension * 2 + 1), dtype=np.int8)
            laplace_data = tf.get_variable("cuda_laplace_matrix", initializer=tf.constant(zeros, dtype=tf.int8),
                                           dtype=tf.int8)

            tf.global_variables_initializer().run()
            laplace = laplace_op.laplace_matrix(dimensions, mask_dimensions, active_mask, fluid_mask, laplace_data,
                                                n ** dimension)

            # Warmup
            print(n)
            print("warmup...")
            print_progress_bar(0, warmup)
            for i in range(warmup):
                zeros = np.zeros(dim_product * (dimension * 2 + 1), dtype=np.int8)
                laplace_data.assign(zeros).eval()

                laplace.eval()
                gc.collect()

                print_progress_bar(i + 1, warmup)

            print_progress_bar(0, testruns)
            gc.collect()
            for i in range(testruns):
                zeros = np.zeros(dim_product * (dimension * 2 + 1), dtype=np.int8)
                laplace_data.assign(zeros).eval()

                start = time.time()
                laplace.eval()
                end = time.time()
                gc.collect()

                testresultsCuda.append(end - start)

                print_progress_bar(i + 1, testruns)

        del fluid_mask, active_mask
        gc.collect()

        testresultsCuda = np.array(testresultsCuda)
        resultCuda.append(testresultsCuda)
        print(n)
        print(np.mean(testresultsCuda))
        print()
        tf.reset_default_graph()
        gc.collect()

    return resultCuda


def benchmark_laplace_matrix_phi(tests, dimension, warmup=10, testruns=50):
    resultPhi = []

    for n in tests:
        dimensions = ([n] * dimension)
        dim_product = n ** dimension
        active_mask = np.ones([1] + [n] * dimension + [1])
        active_mask = np.pad(active_mask, [[0, 0]] + [[1, 1]] * dimension + [[0, 0]], mode='constant')
        fluid_mask = np.ones([1] + [n + 2] * dimension + [1])

        testresultsPhi = []

        # Warmup
        print(n)
        print("warmup...")
        print_progress_bar(0, warmup)
        for i in range(warmup):
            sidx, sorting = sparse_indices(dimensions)
            sval_data = sparse_values(dimensions, active_mask, fluid_mask, sorting)
            A = tf.SparseTensor(indices=sidx, values=sval_data, dense_shape=[dim_product, dim_product])

            del sidx, sorting, sval_data, A
            gc.collect()

            print_progress_bar(i + 1, warmup)

        print_progress_bar(0, testruns)
        gc.collect()
        for i in range(testruns):
            start = time.time()
            sidx, sorting = sparse_indices(dimensions)
            sval_data = sparse_values(dimensions, active_mask, fluid_mask, sorting)
            A = tf.SparseTensor(indices=sidx, values=sval_data, dense_shape=[dim_product, dim_product])
            end = time.time()

            testresultsPhi.append(end - start)

            del sidx, sorting, sval_data, A
            gc.collect()

            print_progress_bar(i + 1, testruns)

        del fluid_mask, active_mask
        gc.collect()

        testresultsPhi = np.array(testresultsPhi)
        resultPhi.append(testresultsPhi)
        print(n)
        print(np.mean(testresultsPhi))
        print()

    return resultPhi
