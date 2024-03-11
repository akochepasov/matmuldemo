#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>

#include <cublas_v2.h>

__global__ void matmul_kernel(int n, float *A, float *B, float *C) {
    float alpha = 1.f, beta = 0.f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(row < n && col < n))
        return;

    float lineSum = 0;
    for (int i = 0; i < n; i++)
        lineSum += A[row * n + i] * B[i * n + col];
    C[row * n + col] = beta * C[row * n + col] + alpha * lineSum;
}

void matmul_cuda(int n, int nthreads, float* A, float* B, float* C) {
    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    int nblocks = n / nthreads;
    dim3 blksPerGrid(nblocks, nblocks);
    dim3 thrsPerBlock(nthreads, nthreads);

    matmul_kernel<<<blksPerGrid, thrsPerBlock>>>(n,
        thrust::raw_pointer_cast(&dvA[0]),
        thrust::raw_pointer_cast(&dvB[0]),
        thrust::raw_pointer_cast(&dvC[0]));

    thrust::copy(dvC.begin(), dvC.end(), C);
}

void matmul_cublas(int n, float* A, float* B, float* C) {
    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    int lda = n, ldb = n, ldc = n;
    const float alpha = 1.0, beta = 0.0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
        thrust::raw_pointer_cast(&dvB[0]), lda,
        thrust::raw_pointer_cast(&dvA[0]), ldb, &beta,
        thrust::raw_pointer_cast(&dvC[0]), ldc);

    thrust::copy(dvC.begin(), dvC.end(), C);
    cublasDestroy(handle);
}
