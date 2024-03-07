#define DEMO_CUDA

#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>


#include "matmuldemo.h"

const int data_align = 64;


static void init_data(int n, float *A, int64_t seed) {
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> dist(1, 100);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = dist(gen);
        }
    }
}

void verify_res(int n, const float* C1, const float* C2, int num) {
    float norm = 0.0;
    float rtol = 1e-04, atol = 1e-04;
    float ntol = 1e-04;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            norm += C1[i * n + j] - C2[i * n + j];
            
            if (abs(C1[i * n + j] - C2[i * n + j]) > atol + rtol * C1[i * n + j]) {
                printf("Error in (%d, %d)\n", i, j);
                printf("%d  - C1:  %f    C2: %f\n", num, C1[i * n + j], C2[i * n + j]);
                throw 1;
            }
        }
    }

    if (norm > ntol) {
        printf("Error: %f\n", norm);
        throw 1;
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

void matmul_omp_simple(int n, const float* A_, const float* B_, float* C_) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;
    const float* A = (const float*)__builtin_assume_aligned(&A_[0], data_align);
    const float* B = (const float*)__builtin_assume_aligned(&B_[0], data_align);
    float* C = (float*)__builtin_assume_aligned(&C_[0], data_align);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float tmpSum = 0.0;
            #pragma omp reduction (+: tmpSum)
            for (int k = 0; k < n; k++) {
                tmpSum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = beta * C[i * n + j] + alpha * tmpSum;
        }
    }
}

void matmul_omp_tile(int n, int ts, const float* A_, const float* B_, float* C_) {
  // C = alpha * A x B + beta * C
  float alpha = 1.0, beta = 0.0;
  const float* A = (const float*)__builtin_assume_aligned(&A_[0], 64);
  const float* B = (const float*)__builtin_assume_aligned(&B_[0], 64);
  float* C       = (float*)      __builtin_assume_aligned(&C_[0], 64);

  omp_set_num_threads(omp_get_num_procs());

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

void matmul_eigen(int n, const float* A_, const float* B_, float* C_) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;
    const float* A = (const float*)__builtin_assume_aligned(&A_[0], 64);
    const float* B = (const float*)__builtin_assume_aligned(&B_[0], 64);
    float*       C =       (float*)__builtin_assume_aligned(&C_[0], 64);

    // "The best code is the code I don't have to write"
    Eigen::Map<const Eigen::MatrixXf> AM(A, n, n);
    Eigen::Map<const Eigen::MatrixXf> BM(B, n, n);
    Eigen::Map<Eigen::MatrixXf> CM(C, n, n);
    CM.noalias() = beta * CM + alpha * (BM * AM); // fortran order!
}

int cnt = 0;

class MatMul : public benchmark::Fixture {
protected:
    static const int num_func = 4;
    int i = 0;
    int n;
    float *A, *B, *C;
    float *D[num_func];

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
        for (int i = 0; i < num_func; i++)
            init_data(n, D[i], (time_t)D[i]);
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
        matmul_naive1(n, A, B, D[0]);
        matmul_naive2(n, A, B, D[1]);
        matmul_omp_simple(n, A, B, D[2]);
        matmul_eigen(n, A, B, D[3]);
    }

    for (int i=1; i < num_func; i++)
        verify_res(n, D[i-1], D[i], i);
}

BENCHMARK_REGISTER_F(MatMul, Verify)
    ->Unit(benchmark::kMillisecond)
    ->Arg(16);


static const int step = 2048;
static const int from = 2048;
static const int nsteps = 1; // 2048 = 32M
static const int to = from + nsteps * step;

#define BENCH_PARAMS_SIMPLE       \
    Unit(benchmark::kMillisecond) \
    ->Arg(benchmark::CreateDenseRange(from, to, step));

#define BENCH_PARAMS_TILED        \
    Unit(benchmark::kMillisecond) \
    ->ArgsProduct({               \
      benchmark::CreateDenseRange(from, to, step),  \
      benchmark::CreateRange(4, 64, /*multi=*/2)    \
    })


#if 0 // Too slow
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

BENCHMARK_DEFINE_F(MatMul, OpenMPSimple)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_omp_simple(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, OpenMPSimple)->BENCH_PARAMS_SIMPLE;

BENCHMARK_DEFINE_F(MatMul, Eign)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_eigen(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, Eign)->BENCH_PARAMS_SIMPLE;

BENCHMARK_DEFINE_F(MatMul, OpenMPTile)(benchmark::State& st) {
    int ts = st.range(1);
    for (auto _ : st) {
        matmul_omp_tile(n, ts, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, OpenMPTile)->
    Unit(benchmark::kMillisecond) 
    ->Args({ 1024, 512 });


BENCHMARK_MAIN();
