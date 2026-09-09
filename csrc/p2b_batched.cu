#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include "util.h"
#include "util.cuh"

#define EXL3_GEMM_ARGS const half*, const uint16_t*, void*, const int, const int, const int, int*, const half*, half*, const half*
template <int, bool, int, int, int, bool>
__device__ void exl3_gemv_kernel(EXL3_GEMM_ARGS);

template <int BITS, int CB>
__global__ __launch_bounds__(512)
void p2b_worklist_kernel(const half* A, const uint16_t** tptrs,
                        const half** suptrs, const half** svptrs,
                        const int32_t* ids, half* C, half* A_had, int* locks,
                        int experts, int size_m, int size_k, int size_n, int stride_a) {
    constexpr int COLS = 32;
    auto grid = cooperative_groups::this_grid();
    const int group = blockIdx.x;
    const int groups = (size_n + COLS - 1) / COLS;
    for (int e = 0; e < experts; ++e) {
        const int idx = ids[e];
        const half* a = (stride_a > 0) ? (A + static_cast<size_t>(e) * size_m * size_k) : A;
        const uint16_t* b = tptrs[idx];
        void* c = C + static_cast<size_t>(e) * size_m * size_n;
        const half* su = suptrs[idx];
        half* ah = A_had + static_cast<size_t>(e) * size_m * size_k;
        const half* sv = svptrs[idx];
        // The QTIP kernel's grid-stride group loop uses blockIdx.x directly.
        // Parent blocks therefore represent column groups; blocks beyond the
        // active group count simply participate in the required barriers.
        exl3_gemv_kernel<BITS, false, CB, 0, 0, false>(
            a, b, c, size_m, size_k, size_n,
            locks + e * (1 << 20), su, ah, sv);
    }
}

// Instantiate the proven QTIP body as a device-callable template.
#define __global__ __device__
#define __launch_bounds__(...)
#include "quant/exl3_gemv_kernel.cuh"
#undef __launch_bounds__
#undef __global__

template <int BITS, int CB>
void launch_batched(const at::Tensor& x, const at::Tensor& tp,
                    const at::Tensor& up, const at::Tensor& vp,
                    const at::Tensor& ids, at::Tensor& out, at::Tensor& ah,
                    at::Tensor& locks, int e, int m, int k, int n, int stride_a) {
    int dev = 0; cudaGetDevice(&dev); int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int resident = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
        p2b_worklist_kernel<BITS, CB>, 512, 0);
    const int groups = n / 32;
    const int grid = std::max(1, std::min(e * groups, resident * sms));
    const half* ap = reinterpret_cast<const half*>(x.data_ptr<c10::Half>());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const uint16_t** tp_ptr = reinterpret_cast<const uint16_t**>(tp.data_ptr<int64_t>());
    const half** up_ptr = reinterpret_cast<const half**>(up.data_ptr<int64_t>());
    const half** vp_ptr = reinterpret_cast<const half**>(vp.data_ptr<int64_t>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<c10::Half>());
    half* ah_ptr = reinterpret_cast<half*>(ah.data_ptr<c10::Half>());
    int* lock_ptr = locks.data_ptr<int>();
    int32_t* id_ptr = ids.data_ptr<int32_t>();
    int experts = e, size_m = m, size_k = k, size_n = n, sa = stride_a;
    void* args[] = {(void*)&ap, (void*)&tp_ptr, (void*)&up_ptr, (void*)&vp_ptr,
                    (void*)&id_ptr, (void*)&out_ptr,
                    (void*)&ah_ptr, (void*)&lock_ptr, (void*)&experts,
                    (void*)&size_m, (void*)&size_k, (void*)&size_n, (void*)&sa};
    cuda_check(cudaLaunchCooperativeKernel(
        (void*)p2b_worklist_kernel<BITS, CB>, dim3(grid), dim3(512), args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor p2b_gemv_batched_cuda(const at::Tensor& x,
                                 const at::Tensor& trellis_ptrs,
                                 const at::Tensor& suh_ptrs,
                                 const at::Tensor& svh_ptrs,
                                 const at::Tensor& expert_indices,
                                 int64_t bits, bool mcg, int64_t mmode) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "batched GEMV requires CUDA fp16");
    TORCH_CHECK(trellis_ptrs.is_cuda() && suh_ptrs.is_cuda() && svh_ptrs.is_cuda() && expert_indices.is_cuda(),
                "all batched arguments must be CUDA tensors");
    const int e = static_cast<int>(expert_indices.numel());
    const int k = static_cast<int>(x.size(-1));
    const int m = (x.dim() == 3) ? static_cast<int>(x.size(1)) : static_cast<int>(x.size(0));
    const int stride_a = (x.dim() == 3) ? 1 : 0;
    const int n = (mmode > 0) ? static_cast<int>(mmode) : 2048;
    auto out = at::empty({e, m, n}, x.options().dtype(at::kHalf));
    auto ah = at::empty({e, m, k}, x.options().dtype(at::kHalf));
    auto locks = at::zeros({e * (1 << 20)}, x.options().dtype(at::kInt));
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "batched GEMV supports K=2,3,4");
    if (bits == 2 && mcg) launch_batched<2, 1>(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, out, ah, locks, e, m, k, n, stride_a);
    else if (bits == 3 && mcg) launch_batched<3, 1>(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, out, ah, locks, e, m, k, n, stride_a);
    else if (bits == 4 && mcg) launch_batched<4, 1>(x, trellis_ptrs, suh_ptrs, svh_ptrs, expert_indices, out, ah, locks, e, m, k, n, stride_a);
    else TORCH_CHECK(false, "batched GEMV currently requires MCG codebook");
    return out;
}
