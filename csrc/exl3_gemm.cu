#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include "exl3_gemv.cuh"

// High-throughput chunked EXL3 GEMM for prompt prefill and batched inference.
// Chunks the batch dimension into optimal power-of-two GEMV slices (up to m=8)
// to maximize SM residency and memory bandwidth saturation without intermediate weight materialization.
at::Tensor exl3_gemm_cuda(const at::Tensor& x, const at::Tensor& trellis,
                          const at::Tensor& suh, const at::Tensor& svh,
                          int64_t bits, bool mcg) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "GEMM requires CUDA fp16 input");
    TORCH_CHECK(x.dim() == 2, "GEMM input must be [m,k]");
    const int64_t m = x.size(0);
    const int64_t n = trellis.size(1) * 16;
    auto out = at::empty({m, n}, x.options());

    int64_t row = 0;
    while (row < m) {
        int64_t chunk = std::min<int64_t>(8, m - row);
        if (chunk > 4 && chunk < 8) chunk = 4;
        else if (chunk == 3) chunk = 2;

        auto x_chunk = x.narrow(0, row, chunk);
        auto y = exl3_gemv_cuda(x_chunk, trellis, suh, svh, bits, mcg, 0);
        out.narrow(0, row, chunk).copy_(y);
        row += chunk;
    }
    return out;
}
