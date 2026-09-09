import json
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, List

import torch
from safetensors.torch import save_file

import sglang_exl3
from sglang_exl3.loader import (
    exl3_mmap_weights_iterator,
    is_unified_memory_platform,
)


def standard_safetensors_iterator(files: List[str], target_device: str):
    import safetensors
    for fpath in files:
        with safetensors.safe_open(fpath, framework="pt", device="cpu") as handle:
            for k in handle.keys():
                t = handle.get_tensor(k)
                if target_device.startswith("cuda"):
                    t = t.to(target_device, non_blocking=True)
                yield k, t


def run_benchmark():
    sglang_exl3.register()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    unified_mem = is_unified_memory_platform()

    tmp_dir = tempfile.mkdtemp(prefix="sglang_exl3_mmap_bench_")
    num_shards = 4
    tensors_per_shard = 16
    tensor_shape = (1024, 1024)
    dtype = torch.float16

    shard_files = []
    total_bytes = 0

    print("Generating synthetic checkpoint shards for mmap benchmark...")
    for i in range(num_shards):
        shard_path = os.path.join(tmp_dir, f"model-{i+1:05d}-of-{num_shards:05d}.safetensors")
        shard_dict = {}
        for j in range(tensors_per_shard):
            key = f"model.layers.{i}.weight_{j}"
            t = torch.randn(tensor_shape, dtype=dtype)
            shard_dict[key] = t
            total_bytes += t.nbytes
        save_file(shard_dict, shard_path)
        shard_files.append(shard_path)

    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 ** 3)

    # Benchmark 1: Standard safetensors CPU -> Device copy
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    std_tensors = 0
    for name, t in standard_safetensors_iterator(shard_files, target_device):
        std_tensors += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    std_elapsed = time.perf_counter() - t0
    std_throughput = (total_gb / std_elapsed) if std_elapsed > 0 else 0.0

    # Benchmark 2: EXL3 Fast MMap Iterator with direct mapping and pre-faulting
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    mmap_tensors = 0
    for name, t in exl3_mmap_weights_iterator(
        shard_files,
        device=target_device,
        num_threads=min(8, os.cpu_count() or 4),
        prefetch_shards=2,
    ):
        mmap_tensors += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mmap_elapsed = time.perf_counter() - t0
    mmap_throughput = (total_gb / mmap_elapsed) if mmap_elapsed > 0 else 0.0

    speedup = (std_elapsed / mmap_elapsed) if mmap_elapsed > 0 else 1.0

    # Project for full DeepSeek-V4 checkpoint (83.4 GiB)
    dsv4_gb = 83.4
    proj_std_sec = dsv4_gb / std_throughput if std_throughput > 0 else 0
    proj_mmap_sec = dsv4_gb / mmap_throughput if mmap_throughput > 0 else 0

    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    results = {
        "device": device_name,
        "unified_memory": unified_mem,
        "total_test_data_mb": round(total_mb, 2),
        "standard_loader": {
            "elapsed_seconds": round(std_elapsed, 4),
            "throughput_gb_s": round(std_throughput, 2),
            "projected_83_4gb_seconds": round(proj_std_sec, 2),
        },
        "exl3_mmap_loader": {
            "elapsed_seconds": round(mmap_elapsed, 4),
            "throughput_gb_s": round(mmap_throughput, 2),
            "projected_83_4gb_seconds": round(proj_mmap_sec, 2),
        },
        "speedup_ratio": round(speedup, 2),
    }

    print("\n================ EXL3 UNIFIED MMAP LOADER BENCHMARK ================")
    print(f"Device: {device_name} (Unified Memory: {unified_mem})")
    print(f"Benchmark Shards: {num_shards} ({total_mb:.1f} MB total)")
    print("--------------------------------------------------------------------")
    print(f" Standard Loader : {std_elapsed*1000:6.1f} ms  | {std_throughput:5.2f} GB/s | Proj DSV4: {proj_std_sec:5.1f}s")
    print(f" EXL3 MMap Loader: {mmap_elapsed*1000:6.1f} ms  | {mmap_throughput:5.2f} GB/s | Proj DSV4: {proj_mmap_sec:5.1f}s")
    print(f" Speedup Ratio   : {speedup:5.2f}x")
    print("====================================================================\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_benchmark()
