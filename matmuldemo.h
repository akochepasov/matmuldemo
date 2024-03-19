#pragma once

#include <iostream>


#ifdef DEMO_CUDA
void matmul_cuda(int n, int bs, float* A, float* B, float* C);
void matmul_cublas(int n, float* A, float* B, float* C);
#endif // DEMO_CUDA

