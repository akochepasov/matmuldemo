// matmuldemo.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <benchmark/benchmark.h>


#include "matmuldemo.h"

static void init_data(int n, float *A, int64_t seed_) {
    int64_t seed = seed_;
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> dist(1, 100);
    //Eigen::VectorXi testVec = Eigen::VectorXi::NullaryExpr(10, [&]() { return dis(gen); });

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

    if (norm > 1e-8)
    {
        printf("Error: %f\n", norm);
        throw 1;
    }
}

static void matmul_naive(int n, const float *A_mat, const float *B_mat, float *C_out) {
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


class MatMul : public benchmark::Fixture {
protected:
    int i = 0;
    int n;
    const int align = 64;
    float *A, *B, *C;
public:
    void SetUp(const ::benchmark::State& state) {
        n = state.range(0);

        A = (float *)_mm_malloc(n * n * sizeof(float), align);
        B = (float *)_mm_malloc(n * n * sizeof(float), align);
        C = (float *)_mm_malloc(n * n * sizeof(float), align);

        init_data(n, A, (time_t)A);
        init_data(n, B, (time_t)B);

        Eigen::Map<Eigen::MatrixXf> matA(A, n, n);
        Eigen::Map<Eigen::MatrixXf> matB(B, n, n);
        Eigen::Map<Eigen::MatrixXf> matC(C, n, n);

        std::cout << "matrix A:" << std::endl;
        std::cout << matA << std::endl;

        std::cout << "matrix B:" << std::endl;
        std::cout << matB << std::endl;

        std::cout << "matrix C:" << std::endl;
        std::cout << matC << std::endl;

    }

    void TearDown(const ::benchmark::State& state) {
        _mm_free(C);
        _mm_free(B);
        _mm_free(A);
    }
};


BENCHMARK_DEFINE_F(MatMul, Verify)(benchmark::State& st) {
    n = st.range(0);

    for (auto _ : st) {
        matmul_naive(n, A, B, C);
        matmul_naive2(n, A, B, C);
    }
}

BENCHMARK_REGISTER_F(MatMul, Verify)
    ->Unit(benchmark::kMillisecond)
    ->Arg(4)
    ->UseRealTime();


BENCHMARK_MAIN();


//int main()
//{
//	std::cout << "Hello CMake." << std::endl;
//	return 0;
//}
