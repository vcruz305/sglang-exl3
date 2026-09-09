#!/usr/bin/env python3
"""Stage Profiler: Measures microsecond timing of Attention, MoE, Linear, and Draft components on GB10."""

import os
import sys
import time
import torch

def profile_components():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    dev = torch.device("cuda:0")
    print(f"Profiling on {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})")
    
    # 1. Measure MoE kernel latency for M=1, M=2, M=3, M=4
    print("\n=== 1. MoE Kernel Latency Across Batch Sizes (43 layers) ===")
    try:
        from sglang_exl3.ops import register_custom_ops
        register_custom_ops()
        from sglang_exl3.exl3 import (
            EXL3Config,
            EXL3MoEMethod,
            _apply_native_fused_moe,
            apply_exl3_fused_moe,
        )

        class DummyMoELayer(torch.nn.Module):
            def __init__(self):
                super().__init__()

        layer = DummyMoELayer()
        cfg = EXL3Config(bits=2)
        method = EXL3MoEMethod(cfg, bits=2)

        num_experts = 256
        hidden = 4096
        inter = 2048
        topk = 6

        print(f"Allocating dummy test layer: {num_experts} experts, hidden={hidden}, inter={inter}...")
        method.create_weights(layer, num_experts, hidden, inter, torch.float16)
        layer.to(dev)
        method.process_weights_after_loading(layer)

        for M in [1, 2, 3, 4, 6]:
            x = torch.randn((M, hidden), dtype=torch.float16, device=dev)
            ids = torch.randint(0, num_experts, (M, topk), dtype=torch.int32, device=dev)
            weights = torch.softmax(torch.randn((M, topk), dtype=torch.float16, device=dev), dim=-1)

            # Test A: Native p2b_fused_moe (row-by-row loop)
            # Warmup
            for _ in range(5):
                _ = _apply_native_fused_moe(x, ids, weights, layer, layer._exl3_inners, None)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            iters = 30
            start_event.record()
            for _ in range(iters):
                _ = _apply_native_fused_moe(x, ids, weights, layer, layer._exl3_inners, None)
            end_event.record()
            torch.cuda.synchronize()

            nat_lat_us = (start_event.elapsed_time(end_event) / iters) * 1000.0
            nat_43_ms = nat_lat_us * 43.0 / 1000.0

            # Test B: Batched apply_exl3_fused_moe (exllamav3_ext)
            fused_lat_us = 0.0
            fused_43_ms = 0.0
            try:
                for _ in range(5):
                    _ = apply_exl3_fused_moe(x, ids, weights, layer, layer._exl3_inners, None, 10.0)
                torch.cuda.synchronize()

                start_event.record()
                for _ in range(iters):
                    _ = apply_exl3_fused_moe(x, ids, weights, layer, layer._exl3_inners, None, 10.0)
                end_event.record()
                torch.cuda.synchronize()

                fused_lat_us = (start_event.elapsed_time(end_event) / iters) * 1000.0
                fused_43_ms = fused_lat_us * 43.0 / 1000.0
            except Exception as e:
                fused_lat_us = -1.0

            print(f"  M={M}: Native row-loop = {nat_lat_us:.1f} us/layer ({nat_43_ms:.2f} ms/43L) | Batched exl3_moe = {fused_lat_us:.1f} us/layer ({fused_43_ms:.2f} ms/43L)")

    except Exception as e:
        print(f"MoE profiling exception: {e}")
        import traceback
        traceback.print_exc()

    # 2. Measure Attention latency via FlashInfer / Triton
    print("\n=== 2. Attention Latency Across Batch Sizes (43 layers) ===")
    try:
        from sglang.kernels.ops.attention.flash_mla_sm120_triton import flash_mla_sparse_decode_triton
        num_heads = 64
        head_dim_v = 512
        num_pages = 200
        page_size = 256
        k_cache = torch.zeros((num_pages, page_size, 1, 584), dtype=torch.uint8, device=dev)
        
        for bs in [1, 2, 3, 4, 6]:
            q = torch.randn((bs, 1, num_heads, 576), dtype=torch.bfloat16, device=dev)
            indices = torch.zeros((bs, 1, 128), dtype=torch.int32, device=dev)
            topk_lens = torch.tensor([128] * bs, dtype=torch.int32, device=dev)
            
            for _ in range(5):
                flash_mla_sparse_decode_triton(q, k_cache, indices, topk_lens, None, head_dim_v, 0.05)
            torch.cuda.synchronize()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            iters = 50
            start_event.record()
            for _ in range(iters):
                flash_mla_sparse_decode_triton(q, k_cache, indices, topk_lens, None, head_dim_v, 0.05)
            end_event.record()
            torch.cuda.synchronize()
            
            lat_per_layer_ms = start_event.elapsed_time(end_event) / iters
            print(f"  BS={bs}: {lat_per_layer_ms:.3f} ms/layer | 43-layer Attention: {lat_per_layer_ms * 43.0:.2f} ms")
    except Exception as e:
        print(f"Attention profiling exception: {e}")

    # 3. Measure Non-Routed Dense Linear GEMM (BF16 vs FP8)
    print("\n=== 3. Dense Linear GEMM Latency Across 43 Layers ===")
    # 6.5B parameters across 43 layers ~= 150M params/layer
    # Main projection is 4096 x 16384 (o_proj) and shared expert 4096 x 2048 x 2
    for bs in [1, 3]:
        x_bf16 = torch.randn(bs, 4096, dtype=torch.bfloat16, device=dev)
        w_bf16 = torch.randn(16384, 4096, dtype=torch.bfloat16, device=dev)
        
        # Warmup BF16
        for _ in range(10):
            _ = torch.matmul(x_bf16, w_bf16.t())
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        iters = 50
        start_event.record()
        for _ in range(iters):
            _ = torch.matmul(x_bf16, w_bf16.t())
        end_event.record()
        torch.cuda.synchronize()
        bf16_ms = start_event.elapsed_time(end_event) / iters
        
        # Test FP8 scaled mm
        x_fp8 = x_bf16.to(torch.float8_e4m3fn)
        w_fp8 = w_bf16.to(torch.float8_e4m3fn)
        scale_x = torch.tensor(1.0, device=dev)
        scale_w = torch.tensor(1.0, device=dev)
        
        fp8_ms = 0.0
        if hasattr(torch, "_scaled_mm"):
            for _ in range(10):
                _ = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=scale_x, scale_b=scale_w, out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            
            start_event.record()
            for _ in range(iters):
                _ = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=scale_x, scale_b=scale_w, out_dtype=torch.bfloat16)
            end_event.record()
            torch.cuda.synchronize()
            fp8_ms = start_event.elapsed_time(end_event) / iters
            
        print(f"  M={bs} (4096 x 16384): BF16={bf16_ms:.3f} ms/layer ({bf16_ms*43:.2f} ms/43-layers) | FP8={fp8_ms:.3f} ms/layer ({fp8_ms*43:.2f} ms/43-layers)")

if __name__ == "__main__":
    profile_components()
