#!/usr/bin/env bash

rm -r ./build/
mkdir ./build/

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

/usr/local/cuda/bin/nvcc -std=c++11 -c -o ./build/laplace_op.cu.o ./src/laplace_op.cu.cc ${TF_CFLAGS[@]} -x cu -Xcompiler -fPIC

# just used for laplace benchmark, can be removed when benchmark is not required
g++ -std=c++11 -shared -o ./build/laplace_op.so ./src/laplace_op.cc ./build/laplace_op.cu.o ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]}

/usr/local/cuda/bin/nvcc -lcublas -std=c++11 -c -o ./build/pressure_solve_op.cu.o ./src/pressure_solve_op.cu.cc ${TF_CFLAGS[@]} -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o ./build/pressure_solve_op.so ./src/pressure_solve_op.cc ./build/pressure_solve_op.cu.o ./build/laplace_op.cu.o ${TF_CFLAGS[@]} -fPIC ${TF_LFLAGS[@]}