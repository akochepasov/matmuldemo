// matmuldemo.cpp : Defines the entry point for the application.
//

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

    if (norm > 1e-8) {
        printf("Error: %f\n", norm);
        throw 1;
    }
}


static void matmul_naive1(int n, const float *A_mat, const float *B_mat, float *C_out) {
    // C = alpha * A x B + beta * C
    const float *A = &A_mat[0];
    const float *B = &B_mat[0];
    float *C = &C_out[0];

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

void matmul_eigen(int n, const float* A_mat, const float* B_mat, float* C_out) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;
    const float* A = (const float*)__builtin_assume_aligned(&A_mat[0], 64);
    const float* B = (const float*)__builtin_assume_aligned(&B_mat[0], 64);
    float*       C =       (float*)__builtin_assume_aligned(&C_out[0], 64);

    // "The best code is the code I don't have to write"
    Eigen::Map<const Eigen::MatrixXf> AM(A, n, n);
    Eigen::Map<const Eigen::MatrixXf> BM(B, n, n);
    Eigen::Map<Eigen::MatrixXf> CM(C, n, n);
    CM.noalias() = beta * CM + alpha * (BM * AM); // fortran order!
}

int cntr = 0;

class MatMul : public benchmark::Fixture {
protected:
    static const int num_func = 3;
    int i = 0;
    int n;
    float *A, *B, *C;
    float *D[num_func];

public:
    void SetUp(const ::benchmark::State& state) {
        n = state.range(0);

        std::cout << "SetUp " << cntr << " :" << n << std::endl;

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

        //std::cout << "TearDown " << cntr++ << std::endl;
    }
};


BENCHMARK_DEFINE_F(MatMul, Verify)(benchmark::State& st) {
    n = st.range(0);

    for (auto _ : st) {
        matmul_naive1(n, A, B, D[0]);
        matmul_naive2(n, A, B, D[1]);
        matmul_eigen(n, A, B, D[2]);
    }

    for (int i=1; i < num_func; i++)
        verify_res(n, D[i-1], D[i], i);
}

BENCHMARK_REGISTER_F(MatMul, Verify)
    ->Unit(benchmark::kMillisecond)
    ->Arg(4)
    ->UseRealTime();


BENCHMARK_DEFINE_F(MatMul, Naive1)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_naive1(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, Naive1)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1000)
    ->UseRealTime();

BENCHMARK_DEFINE_F(MatMul, Eign)(benchmark::State& st) {
    for (auto _ : st) {
        matmul_eigen(n, A, B, C);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_REGISTER_F(MatMul, Eign)
->Unit(benchmark::kMillisecond)
->Arg(1000)
->UseRealTime();


BENCHMARK_MAIN();
