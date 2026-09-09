#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemv_kernel.cuh"

static void* select_kernel(int bits, int cb, int mmode, int cfg) {
    #define SEL(b_, cb_, m_, c_) \
        if (bits == b_ && cb == cb_ && mmode == m_ && cfg == c_) \
            return (void*) exl3_gemv_kernel<b_, false, cb_, m_, c_, false>;
    #define GRID(b_, cb_) \
        SEL(b_, cb_, 0, 0) SEL(b_, cb_, 0, 1) SEL(b_, cb_, 1, 0) SEL(b_, cb_, 1, 1)
    GRID(2, 1) GRID(2, 2) GRID(3, 1) GRID(3, 2)
    GRID(4, 0) GRID(4, 1) GRID(4, 2)
    #undef GRID
    #undef SEL
    return nullptr;
}

at::Tensor exl3_gemv_cuda(const at::Tensor& x, const at::Tensor& trellis,
                          const at::Tensor& suh, const at::Tensor& svh,
                          int64_t bits, bool mcg, int64_t mmode) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "CUDA GEMV requires fp16 input");
    TORCH_CHECK(trellis.is_cuda() && suh.is_cuda() && svh.is_cuda(), "GEMV tensors must share CUDA device");
    const int m = static_cast<int>(x.numel() / x.size(-1));
    const int k = static_cast<int>(x.size(-1));
    const int n = static_cast<int>(trellis.size(1) * 16);
    const int cb = mcg ? 1 : 2;
    const int mode = m == 1 ? 0 : 1;
    const int cfg = 0;
    void* kernel = select_kernel(static_cast<int>(bits), cb, mode, cfg);
    TORCH_CHECK(kernel != nullptr, "unsupported EXL3 GEMV K/codebook configuration");
    auto out = at::empty({m, n}, x.options().dtype(at::kHalf));
    auto a_had = at::empty_like(x);
    auto locks = at::zeros({1 << 20}, x.options().dtype(at::kInt));
    int device = 0;
    cudaGetDevice(&device);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int threads = cfg == 0 ? 512 : 256;
    const int cols = cfg == 0 ? 32 : 64;
    int resident = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, threads, 0));
    const int grid = std::max(1, std::min(n / cols, resident * sms));
    const half* ap = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    const uint16_t* bp = reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>());
    void* cp = out.data_ptr<c10::Half>();
    int* lp = locks.data_ptr<int>();
    const half* sup = reinterpret_cast<const half*>(suh.data_ptr<c10::Half>());
    half* ahp = reinterpret_cast<half*>(a_had.data_ptr<c10::Half>());
    const half* svp = reinterpret_cast<const half*>(svh.data_ptr<c10::Half>());
    void* args[] = {(void*)&ap, (void*)&bp, (void*)&cp, (void*)&m, (void*)&k,
                    (void*)&n, (void*)&lp, (void*)&sup, (void*)&ahp, (void*)&svp};
    cuda_check(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(threads), args, 0,
                                           at::cuda::getCurrentCUDAStream().stream()));
    cuda_check(cudaPeekAtLastError());
    (void)mmode;
    return out;
}
