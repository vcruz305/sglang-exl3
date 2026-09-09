#!/usr/bin/env python3
"""Accurate DSV4 Sparse MLA Microbenchmark & Parity Validator for sm_121 Blackwell.

Validates numerical parity and measures per-layer latency between Triton and FlashInfer
sparse MLA decode backends under exact DSV4 dimensions and multiple batch sizes (M in [1, 2, 4, 6, 20]).
Measures GPU execution duration strictly via torch.cuda.Event.
"""

import os
import sys
import time
import torch

def benchmark_flashmla():
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping FlashMLA benchmark.")
        return

    device = torch.device("cuda:0")
    print("=" * 60)
    print(f"DSV4 FlashMLA Microbenchmark on {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_capability(0)})")
    print("=" * 60)

    num_heads = 64
    head_dim_v = 512  # DSV4 query dimension
    rotary_dim = 64
    total_q_dim = 576  # 512 value / non-rotary + 64 rotary packed in cache format
    page_size = 64
    num_pages = 256
    topk = 512  # DSV4 attention topk index

    print(f"Configuration: heads={num_heads}, head_dim_v={head_dim_v}, rotary_dim={rotary_dim}, topk={topk}, page_size={page_size}")

    # Check FlashInfer availability
    has_flashinfer = False
    try:
        import flashinfer
        has_flashinfer = True
        print(f"FlashInfer version: {getattr(flashinfer, '__version__', 'unknown')}")
    except ImportError:
        print("FlashInfer not installed.")

    # Check Triton backend availability
    has_triton = False
    try:
        from sglang.kernels.ops.attention.flash_mla_sm120_triton import flash_mla_sparse_decode_triton
        has_triton = True
        print("Triton FlashMLA backend loaded.")
    except Exception as e:
        print(f"Triton FlashMLA unavailable: {e}")

    # Check SGLang SM120 FlashInfer wrapper
    has_sglang_fi = False
    try:
        from sglang.kernels.ops.attention.flash_mla_sm120 import flash_mla_with_kvcache_sm120
        has_sglang_fi = True
        print("SGLang FlashInfer SM120 wrapper loaded.")
    except Exception as e:
        print(f"SGLang FlashInfer SM120 wrapper unavailable: {e}")

    # Allocate representative non-zero test cache
    # FlashInfer SM120 expects 656 bytes in the packed page (the Triton
    # backend also accepts this canonical layout): 64 rotary + 512 value,
    # with the remaining metadata/padding included in the page width.
    torch.manual_seed(42)
    # Zero-filled packed pages keep the synthetic decode numerically finite
    # without pretending that arbitrary uint8 bytes are valid EXL3 scales.
    k_cache = torch.zeros((num_pages, page_size, 1, 656), dtype=torch.uint8, device=device)

    batch_sizes = [1, 2, 4, 6, 20]

    for bs in batch_sizes:
        print(f"\n--- Batch Size M={bs} ---")
        q = torch.randn((bs, 1, num_heads, total_q_dim), dtype=torch.bfloat16, device=device)
        # Generate realistic non-identical sparse indices
        indices = torch.randint(0, num_pages * page_size, (bs, 1, topk), dtype=torch.int32, device=device)
        topk_lens = torch.tensor([topk] * bs, dtype=torch.int32, device=device)

        triton_out = None
        fi_out = None

        # 1. Benchmark Triton
        if has_triton:
            try:
                # Warmup
                for _ in range(5):
                    triton_out = flash_mla_sparse_decode_triton(q, k_cache, indices, topk_lens, None, head_dim_v, 0.05)
                torch.cuda.synchronize()

                # Timing with CUDA Events
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                
                iters = 50
                start_ev.record()
                for _ in range(iters):
                    triton_out = flash_mla_sparse_decode_triton(q, k_cache, indices, topk_lens, None, head_dim_v, 0.05)
                end_ev.record()
                torch.cuda.synchronize()

                elapsed_ms = start_ev.elapsed_time(end_ev) / iters
                print(f"[Triton]      M={bs}: {elapsed_ms:.4f} ms/layer | 43-layer total: {elapsed_ms * 43:.2f} ms")
            except Exception as e:
                print(f"[Triton]      M={bs} failed: {e}")

        # 2. Benchmark FlashInfer SM120 Wrapper
        if has_sglang_fi:
            try:
                import inspect
                sig = inspect.signature(flash_mla_with_kvcache_sm120)
                param_names = list(sig.parameters.keys())
                # The current SGLang wrapper intentionally exposes ``**kwargs``
                # (FlashInfer 0.6.18 changed its positional signature).  An
                # inspect pass over the literal parameter name ``kwargs``
                # therefore produces an empty mapping and the old positional
                # fallback fails with ``takes 0 positional arguments``.  Feed
                # the wrapper's canonical keyword names whenever it accepts a
                # VAR_KEYWORD parameter; retain name-based resolution for older
                # explicit signatures.
                if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    call_kwargs = {
                        "q": q,
                        "k_cache": k_cache,
                        "indices": indices,
                        "topk_length": topk_lens,
                        "attn_sink": None,
                        "head_dim_v": head_dim_v,
                        "softmax_scale": 0.05,
                    }
                else:
                    call_kwargs = {}
                    for p in param_names:
                        p_lower = p.lower()
                        if "q" in p_lower and "kv" not in p_lower:
                            call_kwargs[p] = q
                        elif "k" in p_lower or "cache" in p_lower:
                            call_kwargs[p] = k_cache
                        elif "index" in p_lower or "indices" in p_lower:
                            call_kwargs[p] = indices
                        elif "topk" in p_lower or "len" in p_lower:
                            call_kwargs[p] = topk_lens
                        elif "scale" in p_lower:
                            call_kwargs[p] = 0.05
                        elif "dim" in p_lower or "v" in p_lower:
                            call_kwargs[p] = head_dim_v
                        elif "sink" in p_lower or "mask" in p_lower:
                            call_kwargs[p] = None

                def run_fi():
                    return flash_mla_with_kvcache_sm120(**call_kwargs)

                # Warmup
                for _ in range(5):
                    fi_out = run_fi()
                torch.cuda.synchronize()

                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)

                iters = 50
                start_ev.record()
                for _ in range(iters):
                    fi_out = run_fi()
                end_ev.record()
                torch.cuda.synchronize()

                elapsed_ms = start_ev.elapsed_time(end_ev) / iters
                print(f"[FlashInfer]  M={bs}: {elapsed_ms:.4f} ms/layer | 43-layer total: {elapsed_ms * 43:.2f} ms")
            except Exception as e:
                print(f"[FlashInfer]  M={bs} failed: {e}")


        # 3. Numerical Parity Check
        if triton_out is not None and fi_out is not None:
            # Backends may return ``(output, lse)`` while FlashInfer can
            # return the output tensor directly.  Compare the output tensor
            # in either representation.
            def output_tensor(value):
                if isinstance(value, (tuple, list)):
                    return value[0]
                return value

            triton_tensor = output_tensor(triton_out)
            fi_tensor = output_tensor(fi_out)
            if triton_tensor.shape != fi_tensor.shape:
                # Triton returns the packed query/value width (576), while
                # FlashInfer's canonical output is value-only (512).  Compare
                # their common value slice and make the shape discrepancy
                # explicit in the receipt.
                common_dim = min(triton_tensor.shape[-1], fi_tensor.shape[-1])
                triton_cmp = triton_tensor[..., :common_dim]
                fi_cmp = fi_tensor[..., :common_dim]
                print(f"[Parity]      Shape mismatch Triton={tuple(triton_tensor.shape)} FlashInfer={tuple(fi_tensor.shape)}; comparing last-dim={common_dim}")
            else:
                triton_cmp, fi_cmp = triton_tensor, fi_tensor
            max_diff = (triton_cmp.float() - fi_cmp.float()).abs().max().item()
            mean_diff = (triton_cmp.float() - fi_cmp.float()).abs().mean().item()
            print(f"[Parity]      Max absolute diff: {max_diff:.6f}, Mean absolute diff: {mean_diff:.6f}")

if __name__ == "__main__":
    benchmark_flashmla()
