#pragma once

#include <iostream>


#ifdef USE_CUDA
void matmul_cuda1D(int n, int bs, float* A, float* B, float* C);
void matmul_cuda2D(int n, int bs, float* A, float* B, float* C);
void matmul_cublas(int n, float* A, float* B, float* C);
#endif // DEMO_CUDA

