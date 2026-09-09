#!/usr/bin/env python3
"""Scheduler & Host Overhead Profiler for SGLang Decode on NVIDIA DGX Spark GB10.

Measures microsecond latency of:
1. CUDA Graph replay (GPU kernel execution vs Python host call time)
2. CPU-GPU synchronization and tensor staging (.item(), .cpu(), tolist())
3. Logits sampling (FlashInfer vs PyTorch GPU vs Host)
4. Synthetic SGLang decode step pipeline breakdown
"""

import time
import torch
import numpy as np

def profile_scheduler_overhead():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    dev = torch.device("cuda:0")
    print(f"Profiling on {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})")

    # 1. CUDA Graph replay: GPU execution vs Host invocation
    print("\n=== 1. CUDA Graph Replay: Host Call Overhead vs GPU Execution ===")
    vocab_size = 129280
    hidden_size = 4096
    
    # Dummy static tensors
    static_x = torch.randn(1, hidden_size, device=dev, dtype=torch.bfloat16)
    static_w = torch.randn(hidden_size, hidden_size, device=dev, dtype=torch.bfloat16)
    static_out = torch.empty(1, hidden_size, device=dev, dtype=torch.bfloat16)

    # Warmup and capture graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            static_out = torch.matmul(static_x, static_w)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        static_out = torch.matmul(static_x, static_w)
    torch.cuda.synchronize()

    # Measure GPU duration via CUDA Events
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    iters = 100

    start_ev.record()
    for _ in range(iters):
        graph.replay()
    end_ev.record()
    torch.cuda.synchronize()
    gpu_time_us = (start_ev.elapsed_time(end_ev) / iters) * 1000.0

    # Measure Python host wall time for graph.replay() without synchronize
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    t1 = time.perf_counter()
    host_replay_us = ((t1 - t0) / iters) * 1e6

    print(f"  GPU Kernel Execution:      {gpu_time_us:.2f} us")
    print(f"  Python Host Replay Call:   {host_replay_us:.2f} us")
    print(f"  Host Dispatch Overhead:    {max(0.0, host_replay_us - gpu_time_us):.2f} us")

    # 2. CPU-GPU Blocking Calls & Transfers
    print("\n=== 2. CPU-GPU Data Movement & Synchronization ===")
    # Simulate moving token IDs and input positions
    token_id = torch.tensor([105], device=dev, dtype=torch.int32)
    
    # Measure .item() latency (forces GPU-to-CPU sync)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = token_id.item()
    t1 = time.perf_counter()
    item_lat_us = ((t1 - t0) / iters) * 1e6
    print(f"  token_id.item() (Forces CPU-GPU Sync): {item_lat_us:.2f} us")

    # Measure .cpu() transfer
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = token_id.cpu()
    t1 = time.perf_counter()
    cpu_lat_us = ((t1 - t0) / iters) * 1e6
    print(f"  token_id.cpu() latency:                {cpu_lat_us:.2f} us")

    # Measure torch.cuda.synchronize() latency
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    sync_lat_us = ((t1 - t0) / iters) * 1e6
    print(f"  torch.cuda.synchronize() idle latency: {sync_lat_us:.2f} us")

    # 3. Logits Sampling Latency
    print("\n=== 3. Logits Processing & Greedy Sampling ===")
    logits = torch.randn(1, vocab_size, device=dev, dtype=torch.float32)

    # Test GPU argmax + item()
    t0 = time.perf_counter()
    for _ in range(iters):
        next_tok = torch.argmax(logits, dim=-1).item()
    t1 = time.perf_counter()
    sample_gpu_us = ((t1 - t0) / iters) * 1e6
    print(f"  GPU torch.argmax(logits) + .item():    {sample_gpu_us:.2f} us ({sample_gpu_us/1000:.3f} ms)")

    # Test FlashInfer sampling if available
    try:
        from flashinfer.sampling import sampling_from_probs
        probs = torch.softmax(logits, dim=-1)
        uniform_samples = torch.rand(1, device=dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = sampling_from_probs(probs, uniform_samples)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fi_sample_us = ((t1 - t0) / iters) * 1e6
        print(f"  FlashInfer sampling_from_probs:        {fi_sample_us:.2f} us ({fi_sample_us/1000:.3f} ms)")
    except Exception as e:
        print(f"  FlashInfer sampling not available: {e}")

    # 4. End-to-End Synthetic SGLang Decode Step Breakdown
    print("\n=== 4. Synthetic Decode Step Pipeline Breakdown ===")
    # Profile simulated loop
    input_ids_cpu = [105]
    positions_cpu = [42]
    
    t_stage_start = time.perf_counter()
    for _ in range(50):
        # Step A: Input preparation (copy to GPU)
        t_a0 = time.perf_counter()
        cur_pos_gpu = torch.tensor(positions_cpu, device=dev, dtype=torch.int32)
        cur_id_gpu = torch.tensor(input_ids_cpu, device=dev, dtype=torch.int32)
        
        # Step B: Graph Replay
        graph.replay()
        
        # Step C: Logits + Sampling
        next_tok = torch.argmax(logits, dim=-1).item()
        
        # Step D: Python State Update
        input_ids_cpu.append(next_tok)
        positions_cpu[0] += 1
    t_stage_end = time.perf_counter()

    synthetic_step_ms = ((t_stage_end - t_stage_start) / 50) * 1000.0
    print(f"  Synthetic Pipeline Step Time:          {synthetic_step_ms:.3f} ms")
    print(f"  Synthetic Loop Throughput:             {1000.0 / synthetic_step_ms:.1f} tok/s")
    print("================================================================")

if __name__ == "__main__":
    profile_scheduler_overhead()
