#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>

#include <cublas_v2.h>

#include <cutlass/gemm/device/gemm.h>

#include "matmuldemo.h"


__global__ void matmul_kernel1D(int n, float* A, float* B, float* C) {
    float alpha = 1.f, beta = 0.f;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = id / n; int j = id % n;
    // int j = id / n; int i = id % n; // This access pattern 10-100x slower

    C[i * n + j] *= beta;

    for (int k = 0; k < n; k++)
        C[i * n + j] += alpha * A[i * n + k] * B[k * n + j];
}

__global__ void matmul_kernel2D(int n, float *A, float *B, float *C) {
    float alpha = 1.f, beta = 0.f;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(row < n && col < n))
        return;

    float dotProd = 0;
    for (int i = 0; i < n; i++)
        dotProd += A[row * n + i] * B[i * n + col];
    C[row * n + col] = beta * C[row * n + col] + alpha * dotProd;
}

void matmul_cuda1D(int n, int nthreads, float* A, float* B, float* C) {
    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    int nblocks = CEIL_DIV(n, nthreads);

    matmul_kernel1D<<<nblocks * n, nthreads>>>(n,
        thrust::raw_pointer_cast(&dvA[0]),
        thrust::raw_pointer_cast(&dvB[0]),
        thrust::raw_pointer_cast(&dvC[0]));

    thrust::copy(dvC.begin(), dvC.end(), C);
}

void matmul_cuda2D(int n, int nthreads, float* A, float* B, float* C) {
    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    int nblocks = CEIL_DIV(n, nthreads);

    dim3 blksPerGrid(nblocks, nblocks);
    dim3 thrsPerBlock(nthreads, nthreads);

    matmul_kernel2D<<<blksPerGrid, thrsPerBlock>>>(n,
        thrust::raw_pointer_cast(&dvA[0]),
        thrust::raw_pointer_cast(&dvB[0]),
        thrust::raw_pointer_cast(&dvC[0]));

    thrust::copy(dvC.begin(), dvC.end(), C);
}

void matmul_cublas(int n, float* A, float* B, float* C) {
    const float alpha = 1.0, beta = 0.0;

    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    int lda = n, ldb = n, ldc = n;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
        &alpha, dvB.data().get(), ldb, dvA.data().get(), lda,
        &beta, thrust::raw_pointer_cast(&dvC[0]), ldc);

    thrust::copy(dvC.begin(), dvC.end(), C);
    cublasDestroy(handle);
}

void matmul_cutlass(int n, float *A, float *B, float *C) {
    const float alpha = 1.0, beta = 0.0;

    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    using clMajor = cutlass::layout::ColumnMajor;
    using clGemm = cutlass::gemm::device::Gemm<float,     // Data-type of A matrix
                                                clMajor,  // Layout of A matrix
                                                float,    // Data-type of B matrix
                                                clMajor,  // Layout of B matrix
                                                float,    // Data-type of C matrix
                                                clMajor>; // Layout of C matrix

    float *dA = dvA.data().get();
    float *dB = dvB.data().get();
    float *dC = thrust::raw_pointer_cast(&dvC[0]);

    int lda = n, ldb = n, ldc = n;
    clGemm::Arguments args( {n, n, n},      // Gemm dimensions
                            {dB, ldb},      // Tensor-ref for source matrix B
                            {dA, lda},      // Tensor-ref for source matrix A
                            {dC, ldc},      // Tensor-ref for source matrix B
                            {dC, ldc},      // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                            {alpha, beta}); // Scalars used in the Epilogue

    clGemm()(args);

    thrust::copy(dvC.begin(), dvC.end(), C);
}
