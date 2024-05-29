#include <math.h>

//#include <device_launch_parameters.h> // fix intellisense for blockIdx
//#include <cuda_runtime.h>

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

template <const uint32_t BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
    const float* A, const float* B, float beta, float* C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    // if statement is necessary to make things work under tile quantization
    if (!(cRow < M && cCol < N))
        return;

    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
        tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
}

#define SWAP(a, b) \
    { \
        (a) ^= (b); \
        (b) ^= (a); \
        (a) ^= (b); \
    }

__device__ unsigned next_pow2(unsigned value) {
    // From the bit twiddle page
    unsigned above = (value - 1);
    above |= above >> 1;
    above |= above >> 2;
    above |= above >> 4;
    above |= above >> 8;
    above |= above >> 16;
    return ++above;
}

const int BK = 8;
const int TM = 8;
const int TN = 8;
const int BM_ = 128;
const int BN_ = 128;

__global__ void
sgemm2DBlocktiling(int n, const float* A, const float* B, float* C) {
    // Idea taken from https://github.com/siboehm/SGEMM_CUDA/
    // explanation https://siboehm.com/articles/22/CUDA-MMM
    float alpha = 1.0f, beta = 0.0f;
    int M = n, N = n, K = n;

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    auto BM = min(next_pow2(M), BM_); // BM and BN get smaller int
    auto BN = min(next_pow2(N), BN_); // short dimentions

    // A thread is responsible for calculating TM*TN elements in the blocktile
    const int numThreadsBlocktile = (BM * BN) / (TM * TN);  // 128 * 128 / (8 * 8) = 256

    // BN/TN are the number of threads to span a column
    const int threadRow = threadIdx.x / (BN / TN);  // 128 / 8 = 16
    const int threadCol = threadIdx.x % (BN / TN);  // 128 / 8 = 16

    if (!(threadRow < M && threadCol < N))
        return;

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM_ * BK];  // BM * BK = 128 * 8 = 1024
    __shared__ float Bs[BK * BN_];  // BK * BN = 8 * 128 = 2014

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    const int innerRowA = threadIdx.x / BK;  // BK = 8
    const int innerColA = threadIdx.x % BK;  // BK = 8
    // calculates the number of rows of As that are being loaded in a single step
    // by a single block
    const int strideA = max(numThreadsBlocktile / BK, 1);   // 256 / 8 = 32
    const int innerRowB = threadIdx.x / BN;  // BN = 128
    const int innerColB = threadIdx.x % BN;  // BN = 128
    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    const int strideB = max(numThreadsBlocktile / BN, 1);  // 256 / 128

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = { 0.0 };
    // register caches for As and Bs
    float regM[TM] = { 0.0 };   // TM = 8
    float regN[TN] = { 0.0 };   // TN = 8

    // outer-most loop over block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {  // BK = 8
        // populate the SMEM caches
        int loadOffset;
        for (loadOffset = 0; loadOffset < BM; loadOffset += strideA) {  // (128 * 128) / (8 * 8) / 8 = 32
            As[(innerRowA + loadOffset) * BK + innerColA] =             // (BM * BN) / (TM * TN) / BK
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (loadOffset = 0; loadOffset < BK; loadOffset += strideB) {  // (BM * BN) / (TM * TN) / BN = 32
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
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

void matmul_cuda2D_8tiles(int n, float* A, float* B, float* C) {
    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    auto dA = thrust::raw_pointer_cast(&dvA[0]);
    auto dB = thrust::raw_pointer_cast(&dvB[0]);
    auto dC = thrust::raw_pointer_cast(&dvC[0]);

    auto BM = BM_, BN = BN_;
    dim3 gridDim(CEIL_DIV(n, BN), CEIL_DIV(n, BM));  // (BM, BN) = (128, 128)
    dim3 blockDim((BM * BN) / (TM * TN));            // 128 * 128 / (8 * 8) = 256
    sgemm2DBlocktiling<<<gridDim, blockDim>>>(n, dA, dB, dC);

    thrust::copy(dvC.begin(), dvC.end(), C);
}

void matmul_cuda2D_coalesce(int n, float* A, float* B, float* C) {
    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    auto dA = thrust::raw_pointer_cast(&dvA[0]);
    auto dB = thrust::raw_pointer_cast(&dvB[0]);
    auto dC = thrust::raw_pointer_cast(&dvC[0]);

    const int BK = 32;
    int M = n, N = n, K = n;
    float alpha = 1.0f, beta = 0.0f;
    dim3 gridDim(CEIL_DIV(N, BK), CEIL_DIV(M, BK));
    dim3 blockDim((BK * BK));
    sgemm_global_mem_coalesce<BK><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);

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
