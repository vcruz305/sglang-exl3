"""Empirical benchmark comparing RadixAttention Hadamard prefix cache hits vs recompute on sm_121."""

import json
import time
import torch
import sglang_exl3
from sglang_exl3.hadamard_cache import (
    HadamardPrefixCache,
    transform_hadamard_fast,
)


def run_hadamard_cache_benchmark():
    sglang_exl3.register()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = 4096
    seq_lens = [256, 512, 1024, 2048, 4096]
    results = []

    print("\n================ EXL3 RADIX HADAMARD CACHE BENCHMARK ================")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Hidden Dimension: {hidden}")
    print("----------------------------------------------------------------------")
    print(" Seq Len | Recompute (µs) | Cache Hit (µs) | Speedup Ratio | Memory (MB)")
    print("---------+----------------+----------------+---------------+------------")

    cache = HadamardPrefixCache(max_entries=16)

    for seq_len in seq_lens:
        x = torch.randn((seq_len, hidden), dtype=torch.float16, device=device)
        prefix_hash = 0x1000 + seq_len
        layer_id = 0

        # 1. Warmup transform
        for _ in range(10):
            _ = transform_hadamard_fast(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 2. Benchmark Recompute
        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            x_had = transform_hadamard_fast(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        recompute_us = ((time.perf_counter() - t0) / iters) * 1e6

        # Populate cache
        cache.put(prefix_hash, layer_id, x_had)

        # 3. Benchmark Cache Hit
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = cache.get(prefix_hash, layer_id, seq_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cache_us = ((time.perf_counter() - t0) / iters) * 1e6

        speedup = recompute_us / max(cache_us, 1e-3)
        mem_mb = (seq_len * hidden * 2) / (1024 * 1024)

        results.append({
            "seq_len": seq_len,
            "hidden_dim": hidden,
            "recompute_latency_us": round(recompute_us, 2),
            "cache_hit_latency_us": round(cache_us, 2),
            "speedup_ratio": round(speedup, 2),
            "footprint_mb": round(mem_mb, 2),
        })

        print(f" {seq_len:7d} | {recompute_us:14.2f} | {cache_us:14.2f} | {speedup:12.1f}x | {mem_mb:9.2f} MB")

    print("======================================================================\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_hadamard_cache_benchmark()
