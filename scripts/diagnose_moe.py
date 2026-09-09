import os
import sys
import time
import traceback
import torch

def diagnose():
    print("================ EXL3 MOE KERNEL DIAGNOSTIC ================")
    if not torch.cuda.is_available():
        print("CUDA is not available! Exiting.")
        sys.exit(1)
    
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})")
    
    import sglang_exl3
    sglang_exl3.register()
    
    from sglang_exl3.exl3 import (
        _load_native_exl3_ext,
        get_moe_kernel_backend,
        native_moe_kernel_available,
        EXL3Config,
        EXL3MoEMethod,
        apply_exl3_experts,
        _apply_native_fused_moe,
        _native_moe_dimensions_supported,
    )
    
    c_mod = _load_native_exl3_ext()
    print(f"Native extension loaded: {c_mod}")
    print(f"p2b_fused_moe callable: {callable(getattr(c_mod, 'p2b_fused_moe', None))}")
    print(f"p2b_gemv_batched callable: {callable(getattr(c_mod, 'p2b_gemv_batched', None))}")
    print(f"MoE backend configured: {get_moe_kernel_backend()}")
    
    # Test layer setup
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
    
    print(f"\nCreating test layer: hidden={hidden}, inter={inter}, experts={num_experts}, bits=2, topk={topk}...")
    method.create_weights(layer, num_experts, hidden, inter, torch.float16)
    layer.to(dev)
    method.process_weights_after_loading(layer)
    
    print(f"layer._exl3_ptrs created: {bool(getattr(layer, '_exl3_ptrs', None))}")
    print(f"layer._exl3_bits: {getattr(layer, '_exl3_bits', None)}")
    print(f"layer._exl3_k: {getattr(layer, '_exl3_k', None)}")
    
    # Test tensor input: single decode token M=1
    x = torch.randn((1, hidden), dtype=torch.float16, device=dev)
    ids = torch.randint(0, num_experts, (1, topk), dtype=torch.int32, device=dev)
    weights = torch.softmax(torch.randn((1, topk), dtype=torch.float16, device=dev), dim=-1)
    
    print(f"\nEvaluating _native_moe_dimensions_supported: {_native_moe_dimensions_supported(x, layer, layer._exl3_inners)}")
    
    # Direct call to _apply_native_fused_moe
    print("\nCalling _apply_native_fused_moe directly...")
    try:
        native_out = _apply_native_fused_moe(x, ids, weights, layer, layer._exl3_inners, None)
        if native_out is not None:
            print(f"SUCCESS: _apply_native_fused_moe returned tensor with shape {native_out.shape}, dtype {native_out.dtype}")
        else:
            print("FAILED: _apply_native_fused_moe returned None!")
    except Exception as exc:
        print(f"EXCEPTION in _apply_native_fused_moe: {exc}")
        traceback.print_exc()

    # Direct call to apply_exl3_batched_moe_gemv for M=1
    print("\nCalling apply_exl3_batched_moe_gemv directly (M=1)...")
    try:
        from sglang_exl3.exl3 import apply_exl3_batched_moe_gemv
        batched_out = apply_exl3_batched_moe_gemv(x, ids, weights, layer, layer._exl3_inners, None, 10.0)
        if batched_out is not None:
            print(f"SUCCESS: apply_exl3_batched_moe_gemv (M=1) returned tensor with shape {batched_out.shape}, dtype {batched_out.dtype}")
        else:
            print("FAILED: apply_exl3_batched_moe_gemv (M=1) returned None!")
    except Exception as exc:
        print(f"EXCEPTION in apply_exl3_batched_moe_gemv (M=1): {exc}")
        traceback.print_exc()

    # Direct call to apply_exl3_batched_moe_gemv for M=6 (DSpark verification batch)
    print("\nCalling apply_exl3_batched_moe_gemv directly (M=6 DSpark verify batch)...")
    try:
        x6 = torch.randn((6, hidden), dtype=torch.float16, device=dev)
        ids6 = torch.randint(0, num_experts, (6, topk), dtype=torch.int32, device=dev)
        weights6 = torch.softmax(torch.randn((6, topk), dtype=torch.float16, device=dev), dim=-1)
        batched_out6 = apply_exl3_batched_moe_gemv(x6, ids6, weights6, layer, layer._exl3_inners, None, 10.0)
        if batched_out6 is not None:
            print(f"SUCCESS: apply_exl3_batched_moe_gemv (M=6) returned tensor with shape {batched_out6.shape}, dtype {batched_out6.dtype}")
        else:
            print("FAILED: apply_exl3_batched_moe_gemv (M=6) returned None!")
        out6_exp = apply_exl3_experts(x6, ids6, weights6, layer)
        print(f"apply_exl3_experts (M=6) completed: out shape {out6_exp.shape}, last_apply = '{getattr(layer, '_exl3_last_apply', 'unknown')}'")
    except Exception as exc:
        print(f"EXCEPTION in apply_exl3_batched_moe_gemv (M=6): {exc}")
        traceback.print_exc()
        
    # Standard call through apply_exl3_experts
    print("\nCalling apply_exl3_experts (M=1)...")
    try:
        out = apply_exl3_experts(x, ids, weights, layer)
        last_apply = getattr(layer, "_exl3_last_apply", "unknown")
        print(f"apply_exl3_experts completed: out shape {out.shape}, last_apply = '{last_apply}'")
        if hasattr(layer, "_exl3_native_error"):
            print(f"Native error recorded on layer: {layer._exl3_native_error}")
    except Exception as exc:
        print(f"EXCEPTION in apply_exl3_experts: {exc}")
        traceback.print_exc()
        return


    # Benchmark eager latency
    iters = 100
    print(f"\nBenchmarking {iters} iterations of apply_exl3_experts in Eager mode (mode={last_apply})...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = apply_exl3_experts(x, ids, weights, layer)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    us_per_layer = (elapsed / iters) * 1e6
    ms_43_layers = (elapsed / iters) * 43 * 1000
    proj_tps = 1000.0 / ms_43_layers if ms_43_layers > 0 else 0
    
    print(f"Eager Latency per MoE layer: {us_per_layer:.2f} us")
    print(f"Projected 43-layer MoE time per token: {ms_43_layers:.2f} ms")
    print(f"Projected MoE throughput limit: {proj_tps:.2f} tok/s")

    # Benchmark CUDA graph replay
    print(f"\nTesting CUDA Graph capture on apply_exl3_experts...")
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(device=dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = apply_exl3_experts(x, ids, weights, layer)
        torch.cuda.current_stream(device=dev).wait_stream(s)
        
        with torch.cuda.graph(g, stream=s):
            out_graph = apply_exl3_experts(x, ids, weights, layer)
        torch.cuda.synchronize()
        print("CUDA Graph capture SUCCESSFUL!")
        
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        elapsed_g = time.perf_counter() - t0
        us_graph = (elapsed_g / iters) * 1e6
        ms_43_graph = (elapsed_g / iters) * 43 * 1000
        proj_tps_graph = 1000.0 / ms_43_graph if ms_43_graph > 0 else 0
        print(f"CUDA Graph Latency per MoE layer: {us_graph:.2f} us")
        print(f"Projected 43-layer MoE time in Graph: {ms_43_graph:.2f} ms")
        print(f"Projected Graph MoE throughput limit: {proj_tps_graph:.2f} tok/s")
    except Exception as exc:
        print(f"CUDA Graph capture FAILED: {exc}")
        traceback.print_exc()

    print("============================================================\n")

if __name__ == "__main__":
    diagnose()
