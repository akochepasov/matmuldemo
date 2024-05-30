
#include <torch/torch.h>
#include <torch/script.h>

// libtorch don't work with const
// Moreover, it crashes compiler when combined in one compilation unit with other libs

void matmul_torch_cuda(int n, float *A, float *B, float *C) {
    // C = alpha * A x B + beta * C
    float alpha = 1.0, beta = 0.0;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // "The best code is the code I don't have to write"
    torch::Tensor matAh = torch::from_blob(A, { n, n }, options);
    torch::Tensor matBh = torch::from_blob(B, { n, n }, options);
    torch::Tensor matCh = torch::from_blob(C, { n, n }, options);
    (void)torch::Device(torch::kCUDA); // torch::kCPU requires MKL AVX dll

    torch::Tensor matAd = matAh.to(torch::kCUDA);
    torch::Tensor matBd = matBh.to(torch::kCUDA);
    torch::Tensor matCd = matCh.to(torch::kCUDA);

    torch::addmm_out(matCd, matCd, matAd, matBd, beta, alpha);

    matCh = matCd.to(torch::kCPU);
    std::memcpy(C, matCh.data_ptr(), sizeof(float) * matCh.numel());
}
