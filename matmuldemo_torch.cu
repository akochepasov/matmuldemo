
#include <torch/torch.h>
#include <torch/script.h>

// libtorch don't work with const
// Moreover, it crashes compiler when combined in one compilation unit with other libs

void matmul_torch_cuda(int n, float* A, float* B, float* C) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // "The best code is the code I don't have to write"
    torch::Tensor mat1h = torch::from_blob(A, { n, n }, options);
    torch::Tensor mat2h = torch::from_blob(B, { n, n }, options);
    (void)torch::Device(torch::kCUDA); // torch::kCPU requires MKL AVX dll

    torch::Tensor mat1d = mat1h.to(torch::kCUDA);
    torch::Tensor mat2d = mat2h.to(torch::kCUDA);

    torch::Tensor mat3d = torch::mm(mat1d, mat2d);
    torch::Tensor mat3h = mat3d.to(torch::kCPU);

    std::memcpy(C, mat3h.data_ptr(), sizeof(float) * mat3h.numel());
}
