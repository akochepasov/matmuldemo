
#define USE_CUDA
#define USE_NAIVE // too slow
#define USE_INTRINSICS
#define USE_OPENMP

#include <iostream>
#include <random>

#include <omp.h>
#include <immintrin.h>

#include <Eigen/Dense>

#include <benchmark/benchmark.h>

#include "matmuldemo.h"

const int data_align = 64;


static void init_data(int n, float *A, int64_t seed) {
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> dist(1, 100);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = dist(gen);
}

void verify_res(int n, const float *C1, const float *C2, int mm_id) {
    float totl = 0.0;
    const float rtol = 1e-04f, atol = 1e-04f;
    const float ttol = 1e-01f;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            auto diff = C1[i * n + j] - C2[i * n + j];
            totl += diff;

            if (abs(diff) > atol + rtol * C1[i * n + j]) {
                printf("Error in (%d, %d)\n", i, j);
                printf("%d  - C1:  %f    C2: %f\n", mm_id, C1[i * n + j], C2[i * n + j]);
                throw 1;
            }
        }
    }

    totl = totl / n;  // Average error per line
    if (totl > ttol) {
        printf("Total error (%d): %f\n", mm_id, totl);
        throw "Total error";
    }
}


static void matmul_naive1(int n, const float *A_, const float *B_, float *C_) {
    // C = alpha * A x B + beta * C
    const float *A = &A_[0];
    const float *B = &B_[0];
    float *C = &C_[0];

    float alpha = 1.0, beta = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] *= beta;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += alpha * A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void matmul_naive2(int n, const float *A, const float *B, float *C) {
    // C = alpha * A x B + beta * C

    float alpha = 1.0, beta = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] *= beta;

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] += alpha * A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void matmul_sse(int n, const float *A_, const float *B_, float *C_) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;
    const float *A = (const float *)__builtin_assume_aligned(&A_[0], data_align);
    const float *B = (const float *)__builtin_assume_aligned(&B_[0], data_align);
    float *C = (float *)__builtin_assume_aligned(&C_[0], data_align);

    __m128 alpha4 = _mm_set1_ps(alpha);
    __m128 beta4 = _mm_set1_ps(beta);

    for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j += 4) {
        __m128 c4 = _mm_load_ps(&C[i * n + j]);
        c4 = _mm_mul_ps(beta4, c4);
        _mm_store_ps(&C[i * n + j], c4);
    }

    for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++) {
        __m128 a4 = _mm_set1_ps(A[i * n + k]);
        a4 = _mm_mul_ps(alpha4, a4);
        for (int j = 0; j < n; j += 4) {
            __m128 c4 = _mm_load_ps(&C[i * n + j]);
            __m128 b4 = _mm_load_ps(&B[k * n + j]);
            c4 = _mm_add_ps(_mm_mul_ps(a4, b4), c4);
            _mm_store_ps(&C[i * n + j], c4);
        }
    }
}

void matmul_avx(int n, const float *A_, const float *B_, float *C_) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;
    const float *A = (const float *)__builtin_assume_aligned(&A_[0], data_align);
    const float *B = (const float *)__builtin_assume_aligned(&B_[0], data_align);
    float *C = (float *)__builtin_assume_aligned(&C_[0], data_align);

    __m256 alpha8 = _mm256_set1_ps(alpha);
    __m256 beta8 = _mm256_set1_ps(beta);

    for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j += 8) {
        __m256 c8 = _mm256_load_ps(&C[i * n + j]);
        c8 = _mm256_mul_ps(beta8, c8);
        _mm256_store_ps(&C[i * n + j], c8);
    }

    for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++) {
        __m256 a8 = _mm256_set1_ps(A[i * n + k]);
        a8 = _mm256_mul_ps(alpha8, a8);
        for (int j = 0; j < n; j += 8) {
            __m256 c8 = _mm256_load_ps(&C[i * n + j]);
            __m256 b8 = _mm256_load_ps(&B[k * n + j]);
            c8 = _mm256_add_ps(_mm256_mul_ps(a8, b8), c8);
            _mm256_store_ps(&C[i * n + j], c8);
        }
    }
}

void matmul_omp_simple(int n, const float* A_, const float* B_, float* C_) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;
    const float *A = (const float *)__builtin_assume_aligned(&A_[0], data_align);
    const float *B = (const float *)__builtin_assume_aligned(&B_[0], data_align);
    float       *C =       (float *)__builtin_assume_aligned(&C_[0], data_align);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i * n + j] *= beta;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++)
    for (int j = 0; j < n; j++)
        C[i*n+j] += alpha*A[i*n+k]*B[k*n+j];
}

void matmul_omp_tile(int n, int ts, const float *A_, const float *B_, float *C_) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;
    const float *A = (const float *)__builtin_assume_aligned(&A_[0], data_align);
    const float *B = (const float *)__builtin_assume_aligned(&B_[0], data_align);
    float       *C =       (float *)__builtin_assume_aligned(&C_[0], data_align);

    #pragma omp parallel for collapse(2)
    for(int i=0; i<n; i++)
    for(int j=0; j<n; j++)
        C[i*n+j] *= beta;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i+=ts)
    for (int k = 0; k < n; k+=ts)
    for (int j = 0; j < n; j+=ts)
        for (int ii = i; ii < i+ts; ii++)
        for (int kk = k; kk < k+ts; kk++)
        for (int jj = j; jj < j+ts; jj++)
            C[ii*n+jj] += alpha*A[ii*n+kk]*B[kk*n+jj];
}

void matmul_eigen(int n, const float *A_, const float *B_, float *C_) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;
    const float *A = (const float *)__builtin_assume_aligned(&A_[0], data_align);
    const float *B = (const float *)__builtin_assume_aligned(&B_[0], data_align);
    float       *C =       (float *)__builtin_assume_aligned(&C_[0], data_align);

    // "The best code is the code I don't have to write"
    Eigen::Map<const Eigen::MatrixXf> AM(A, n, n);
    Eigen::Map<const Eigen::MatrixXf> BM(B, n, n);
    Eigen::Map<Eigen::MatrixXf> CM(C, n, n);
    CM.noalias() = beta * CM + alpha * (BM * AM); // fortran order!
}

int cnt = 0;

class MatMul : public benchmark::Fixture {
protected:
    static const int num_func = 12;
    int i = 0;
    int n;
    float *A, *B, *C;
    float *D[num_func]; // output for verification

public:
    void SetUp(const ::benchmark::State& state) {
        n = state.range(0);

        A = (float *)_mm_malloc(n * n * sizeof(float), data_align);
        B = (float *)_mm_malloc(n * n * sizeof(float), data_align);
        C = (float *)_mm_malloc(n * n * sizeof(float), data_align);
        for (int i = 0; i < num_func; i++)
            D[i] = (float*)_mm_malloc(n * n * sizeof(float), data_align);

        init_data(n, A, (time_t)A);
        init_data(n, B, (time_t)B);
    }

    void TearDown(const ::benchmark::State& state) {
        for (int i = 0; i < num_func; i++)
            _mm_free(D[i]);
        _mm_free(C);
        _mm_free(B);
        _mm_free(A);
    }
};


BENCHMARK_DEFINE_F(MatMul, Verify)(benchmark::State& st) {
    n = st.range(0);

    for (auto _ : st) {
        int i = 0; matmul_eigen(n, A, B, D[0]); // Reference function
#ifdef USE_NAIVE
        i++; matmul_naive1(n, A, B, D[i]); verify_res(n, D[0], D[i], i);
        i++; matmul_naive2(n, A, B, D[i]); verify_res(n, D[0], D[i], i);
#endif // USE_NAIVE
#ifdef USE_OPENMP
        i++; matmul_omp_simple(n, A, B, D[i]);  verify_res(n, D[0], D[i], i);
        i++; matmul_omp_tile(n, 4, A, B, D[i]); verify_res(n, D[0], D[i], i);
#endif // USE_OPENMP
#ifdef USE_CUDA
        i++; matmul_cuda1D(n, 4, A, B, D[i]); verify_res(n, D[0], D[i], i);
        i++; matmul_cuda2D(n, 4, A, B, D[i]); verify_res(n, D[0], D[i], i);
        i++; matmul_cuda2D_coalesce(n, A, B, D[i]); verify_res(n, D[0], D[i], i);
        i++; matmul_cublas(n, A, B, D[i]);  verify_res(n, D[0], D[i], i);
        i++; matmul_cutlass(n, A, B, D[i]); verify_res(n, D[0], D[i], i);
#endif // USE_CUDA
#ifdef USE_INTRINSICS
        i++; matmul_sse(n, A, B, D[i]); verify_res(n, D[0], D[i], i);
        i++; matmul_avx(n, A, B, D[i]); verify_res(n, D[0], D[i], i);
#endif // USE_INTRINSICS
    }
}

BENCHMARK_REGISTER_F(MatMul, Verify)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Arg(64);


static const int step = 1024;
static const int from = 1024 + 512; // 2048
static const int nsteps = 1; // 2048 = 32M
static const int to = from + nsteps * step;

#define BENCH_PARAMS_SIMPLE       \
    Unit(benchmark::kMillisecond) \
    ->DenseRange(from, to, step)

#define BENCH_PARAMS_TILED        \
    Unit(benchmark::kMillisecond) \
    ->ArgsProduct({               \
        benchmark::CreateDenseRange(from, to, step),  \
        benchmark::CreateRange(4, 256, /*multi=*/4)   \
    }) // 4, 16, 64, 256


#ifdef USE_NAIVE
BENCHMARK_DEFINE_F(MatMul, Naive1)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_naive1(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, Naive1)->BENCH_PARAMS_SIMPLE;

BENCHMARK_DEFINE_F(MatMul, Naive2)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_naive2(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, Naive2)->BENCH_PARAMS_SIMPLE;
#endif // USE_NAIVE

#ifdef USE_INTRINSICS
BENCHMARK_DEFINE_F(MatMul, SSE)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_sse(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, SSE)->BENCH_PARAMS_SIMPLE;

BENCHMARK_DEFINE_F(MatMul, AVX)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_avx(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, AVX)->BENCH_PARAMS_SIMPLE;
#endif // USE_INTRINSICS

#ifdef USE_OPENMP
BENCHMARK_DEFINE_F(MatMul, OpenMPSimple)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_omp_simple(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, OpenMPSimple)->BENCH_PARAMS_SIMPLE;

BENCHMARK_DEFINE_F(MatMul, OpenMPTile)(benchmark::State& st) {
    int ts = n / st.range(1);
    for (auto _ : st) {
        matmul_omp_tile(n, ts, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, OpenMPTile)->BENCH_PARAMS_TILED;
#endif // USE_OPENMP

BENCHMARK_DEFINE_F(MatMul, Eign)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_eigen(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, Eign)->BENCH_PARAMS_SIMPLE;

#ifdef USE_CUDA
BENCHMARK_DEFINE_F(MatMul, CudaKernel1D)(benchmark::State& st) {
    int thrs = n / st.range(1);
    for (auto _ : st) {
        matmul_cuda1D(n, thrs, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, CudaKernel1D)->BENCH_PARAMS_TILED;

BENCHMARK_DEFINE_F(MatMul, CudaKernel2D)(benchmark::State& st) {
    int thrs = n / st.range(1);
    for (auto _ : st) {
        matmul_cuda2D(n, thrs, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, CudaKernel2D)->BENCH_PARAMS_TILED;

BENCHMARK_DEFINE_F(MatMul, CudaKernelCoalesce)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_cuda2D_coalesce(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, CudaKernelCoalesce)->BENCH_PARAMS_SIMPLE;

//BENCHMARK_DEFINE_F(MatMul, CudaKernel2DTile)(benchmark::State& st) {
//    int thrs = n / st.range(1);
//    for (auto _ : st) {
//        matmul_cuda2D_tile(n, thrs, A, B, C);
//        benchmark::DoNotOptimize(C);
//        benchmark::ClobberMemory();
//    }
//}
//BENCHMARK_REGISTER_F(MatMul, CudaKernel2DTile)->BENCH_PARAMS_TILED;

BENCHMARK_DEFINE_F(MatMul, CuBlas)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_cublas(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, CuBlas)->BENCH_PARAMS_SIMPLE;

BENCHMARK_DEFINE_F(MatMul, Cutlass)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_cutlass(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, Cutlass)->BENCH_PARAMS_SIMPLE;
#endif // USE_CUDA


BENCHMARK_MAIN();
