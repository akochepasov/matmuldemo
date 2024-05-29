#pragma once

#define USE_NAIVE // too slow
#define USE_INTRINSICS
#define USE_OPENMP
#define USE_CUDA

const int data_align = 4096;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#ifdef USE_CUDA
void matmul_cuda1D(int n, int bs, float* A, float* B, float* C);
void matmul_cuda2D(int n, int bs, float* A, float* B, float* C);
void matmul_cuda2D_8tiles(int n, float* A, float* B, float* C);
void matmul_cuda2D_coalesce(int n, float* A, float* B, float* C);
void matmul_cublas(int n, const float* A, const float* B, const float* C);
void matmul_cutlass(int n, const float* A, const float* B, const float* C);
void matmul_torch_cuda(int n, float* A, float* B, float* C);
#endif // DEMO_CUDA
