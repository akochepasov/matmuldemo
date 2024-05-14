#include <math.h>

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

const int BK = 8;
const int TM = 8;
const int TN = 8;
const int BM = 128;
const int BN = 128;

__device__ static bool debug = true;

__global__ void 
sgemm2DBlocktiling(int M, int N, int K, float alpha, const float* A,
    const float* B, float beta, float* C) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int totalResultsBlocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
    assert(numThreadsBlocktile == blockDim.x);

    // BN/TN are the number of threads to span a column
    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    if (!(threadRow < M && threadCol < N))
        return;

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    // calculates the number of rows of As that are being loaded in a single step
    // by a single block
    const int strideA = numThreadsBlocktile / BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    const int strideB = numThreadsBlocktile / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = { 0.0 };
    // register caches for As and Bs
    float regM[TM] = { 0.0 };
    float regN[TN] = { 0.0 };

    // outer-most loop over block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        if (debug) 
            printf( "{%2d %2d: %3d} %d "
                    "As %3d .. %3d (%d) <= A: %5d .. %5d (%d) -> "
                    "Bs %3d .. %3d (%d) <= B: %5d .. %5d (%d)\n",
            cRow, cCol, threadIdx.x, bkIdx,
                ((innerRowA + 0) * BK) + innerColA, ((innerRowA + (BM - 1)) * BK) + innerColA, strideA,
                ((innerRowA + 0) * K ) + innerColA, ((innerRowA + (BM - 1)) * K ) + innerColA, strideA,
                ((innerRowB + 0) * BN) + innerColB, ((innerRowB + (BK - 1)) * BN) + innerColB, strideB,
                ((innerRowB + 0) * N ) + innerColB, ((innerRowB + (BK - 1)) * N ) + innerColB, strideB);
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
            if (debug)
                printf( "{%2d %2d: %3d} %d " 
                        "RegM: %5d .. %5d -> RegN: %5d .. %5d\n", 
                    cRow, cCol, threadIdx.x, dotIdx,
                        (threadRow * TM + 0)    * BK + dotIdx, (threadRow * TM + TM) * BK + dotIdx,
                        dotIdx * BN + threadCol * TN + 0,       dotIdx * BN + threadCol * TN + TN);
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
            if (debug)
                printf( "{%2d %2d: %3d} %d threadRes2D: %5d .. %5d -> threadRes2D: %5d .. %5d\n",
                    cRow, cCol, threadIdx.x, dotIdx,
                               0 * TN + 0,       0 * TN + TN - 1,
                        (TM - 1) * TN + 0, (TM - 1)* TN + TN - 1);
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
    if (debug)
        printf( "{%2d %2d: %3d} C2D: %5d .. %5d -> C2D: %5d .. %5d\n",
        cRow, cCol, threadIdx.x, 
            (threadRow * TM + 0)      * N + threadCol * TN + 0, (threadRow * TM + 0) * N      + threadCol * TN + TN - 1,
            (threadRow * TM + TM - 1) * N + threadCol * TN + 0, (threadRow * TM + TM - 1) * N + threadCol * TN + TN - 1);
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

void matmul_cuda2D_tile(int n, int nthreads, float* A, float* B, float* C) {
    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    //int nblocks = CEIL_DIV(n, nthreads);
    //dim3 blksPerGrid(nblocks, nblocks);
    //dim3 thrsPerBlock(nthreads);
    //dim3 thrsPerBlock(128 * 128 / 8 * 8);

    //matmul_kernel2DBlocktiling<<<blksPerGrid, thrsPerBlock>>> (n,
    //    thrust::raw_pointer_cast(&dvA[0]),
    //    thrust::raw_pointer_cast(&dvB[0]),
    //    thrust::raw_pointer_cast(&dvC[0]));
    auto dA = thrust::raw_pointer_cast(&dvA[0]);
    auto dB = thrust::raw_pointer_cast(&dvB[0]);
    auto dC = thrust::raw_pointer_cast(&dvC[0]);

    int M = n, N = n, K = n;
    float alpha = 1.0f, beta = 0.0f;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);

    thrust::copy(dvC.begin(), dvC.end(), C);
}

void matmul_cuda2D_coalesce(int n, float* A, float* B, float* C) {
    thrust::device_vector<float> dvA(A, A + n * n);
    thrust::device_vector<float> dvB(B, B + n * n);
    thrust::device_vector<float> dvC(n * n);

    auto dA = thrust::raw_pointer_cast(&dvA[0]);
    auto dB = thrust::raw_pointer_cast(&dvB[0]);
    auto dC = thrust::raw_pointer_cast(&dvC[0]);

    int M = n, N = n, K = n;
    float alpha = 1.0f, beta = 0.0f;
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
    dim3 blockDim((32 * 32));
    sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);

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
