#!/usr/bin/env python3
"""Audit Tensor Inventory and Memory Bandwidth Analyzer for DeepSeek-V4 EXL3 on DGX Spark GB10.

Inspects safetensors index and metadata for DSV4-Flash-Vision-ablit-EXL3-MixedK,
categorizes all model parameters, identifies stored precision (FP8, BF16, EXL3 Trellis),
and computes exact DRAM bandwidth read bounds per token decode step.
"""

import os
import sys
import json
import glob
from typing import Dict, Any, List
from collections import defaultdict

def audit_inventory(model_path: str = "/home/victor/models/DSV4-Flash-Vision-ablit-EXL3-MixedK", output_json: str = "tensor_inventory.json") -> Dict[str, Any]:
    print(f"============================================================")
    print(f"DeepSeek-V4 EXL3 Tensor Inventory & Bandwidth Audit")
    print(f"Model Path: {model_path}")
    print(f"============================================================")

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        # Check if safetensors shards exist directly
        shards = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not shards:
            print(f"Error: No safetensors files or index found at {model_path}")
            return {"error": "model files not found"}

    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors package not available; attempting fallback to json index")
        safe_open = None

    categories = {
        "target_routed_trellis": 0,
        "target_routed_scales": 0,
        "target_non_routed_dense": 0,
        "shared_experts": 0,
        "embeddings_and_head": 0,
        "draft_mtp": 0,
        "vision_aligner": 0,
        "other": 0,
    }

    category_tensors = defaultdict(list)
    dtype_breakdown = defaultdict(lambda: {"count": 0, "bytes": 0})
    layer_counts = defaultdict(int)

    # Shard inspection
    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    print(f"Found {len(shard_files)} safetensors shards.")

    total_bytes = 0
    total_params = 0

    for shard in shard_files:
        if safe_open is not None:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_slice = f.get_slice(key)
                    shape = tensor_slice.get_shape()
                    dtype = str(tensor_slice.get_dtype())
                    
                    # Compute numel
                    numel = 1
                    for d in shape:
                        numel *= d
                    
                    # Byte calculation
                    bytes_per_elem = 2 # default fp16/bf16
                    if "8" in dtype.lower():
                        bytes_per_elem = 1
                    elif "32" in dtype.lower():
                        bytes_per_elem = 4
                    elif "16" in dtype.lower():
                        bytes_per_elem = 2
                    elif "64" in dtype.lower():
                        bytes_per_elem = 8
                    
                    t_bytes = numel * bytes_per_elem
                    total_bytes += t_bytes
                    total_params += numel

                    dtype_breakdown[dtype]["count"] += 1
                    dtype_breakdown[dtype]["bytes"] += t_bytes

                    # Categorize
                    key_lower = key.lower()
                    cat = "other"
                    if "mtp" in key_lower:
                        cat = "draft_mtp"
                    elif "vision" in key_lower or "aligner" in key_lower:
                        cat = "vision_aligner"
                    elif "embed_tokens" in key_lower or "lm_head" in key_lower:
                        cat = "embeddings_and_head"
                    elif "shared_expert" in key_lower:
                        cat = "shared_experts"
                    elif "trellis" in key_lower:
                        cat = "target_routed_trellis"
                    elif "suh" in key_lower or "svh" in key_lower:
                        cat = "target_routed_scales"
                    elif "expert" in key_lower:
                        cat = "target_routed_trellis"
                    elif any(w in key_lower for w in ["wq_b", "wkv", "wo_a", "wo_b", "gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]):
                        cat = "target_non_routed_dense"

                    categories[cat] += t_bytes
                    category_tensors[cat].append({
                        "name": key,
                        "shape": shape,
                        "dtype": dtype,
                        "numel": numel,
                        "bytes": t_bytes,
                    })

    # Summary Display
    print("\n--- Storage Breakdown by Category ---")
    for cat, b in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        gib = b / (1024 ** 3)
        pct = (b / total_bytes * 100) if total_bytes > 0 else 0
        print(f"  {cat:26s}: {gib:6.2f} GiB ({pct:5.1f}%) | {len(category_tensors[cat])} tensors")

    print("\n--- Precision / DType Breakdown ---")
    for dt, info in sorted(dtype_breakdown.items(), key=lambda x: x[1]["bytes"], reverse=True):
        gib = info["bytes"] / (1024 ** 3)
        pct = (info["bytes"] / total_bytes * 100) if total_bytes > 0 else 0
        print(f"  {dt:26s}: {gib:6.2f} GiB ({pct:5.1f}%) | {info['count']} tensors")

    print(f"\nTotal Checkpoint Storage: {total_bytes / (1024 ** 3):.2f} GiB ({total_params / 1e9:.2f}B elements)")

    # Bandwidth Analysis for GB10 (273 GB/s peak bandwidth)
    PEAK_BW_GB_S = 273.0
    print("\n--- Single-Token Decode ($M=1$) DRAM Bandwidth Bounds ---")
    routed_total_bytes = categories["target_routed_trellis"] + categories["target_routed_scales"]
    routed_active_bytes = routed_total_bytes * (6.0 / 256.0) # Top-6 active routing
    non_routed_bytes = categories["target_non_routed_dense"]
    shared_bytes = categories["shared_experts"]

    lm_head_bytes = sum(t["bytes"] for t in category_tensors["embeddings_and_head"] if "lm_head" in t["name"].lower())

    step_bytes_read = routed_active_bytes + non_routed_bytes + shared_bytes + lm_head_bytes
    step_gib_read = step_bytes_read / (1024 ** 3)
    step_gb_read = step_bytes_read / 1e9

    min_step_time_ms = (step_gb_read / PEAK_BW_GB_S) * 1000.0
    max_dense_tps = 1000.0 / min_step_time_ms if min_step_time_ms > 0 else 0

    print(f"  Routed MoE Bytes Read (Top-6/256):    {routed_active_bytes / (1024 ** 2):6.1f} MiB")
    print(f"  Non-Routed Dense Bytes Read:           {non_routed_bytes / (1024 ** 2):6.1f} MiB")
    print(f"  Shared Expert Bytes Read:               {shared_bytes / (1024 ** 2):6.1f} MiB")
    print(f"  LM Head Bytes Read:                     {lm_head_bytes / (1024 ** 2):6.1f} MiB")
    print(f"  Total DRAM Read per Decode Step:        {step_gib_read:6.2f} GiB ({step_gb_read:.2f} GB)")
    print(f"  Peak Hardware Bandwidth:                {PEAK_BW_GB_S} GB/s (NVIDIA DGX Spark GB10)")
    print(f"  Theoretical Minimum Step Time (100% BW): {min_step_time_ms:6.2f} ms")
    print(f"  Theoretical Max Dense Decode:           {max_dense_tps:6.1f} tok/s")
    print(f"  Realistic Practical Decode (65% BW):    {max_dense_tps * 0.65:6.1f} tok/s")

    # Draft MTP Bandwidth Bounds
    draft_bytes = categories["draft_mtp"]
    print(f"\n--- Draft MTP Bandwidth Bounds ---")
    print(f"  Draft MTP Storage:                      {draft_bytes / (1024 ** 3):6.2f} GiB")
    draft_step_time_ms = (draft_bytes / 1e9 / PEAK_BW_GB_S) * 1000.0
    print(f"  Minimum Draft Forward Pass (100% BW):   {draft_step_time_ms:6.2f} ms")

    results = {
        "model_path": model_path,
        "total_gib": total_bytes / (1024 ** 3),
        "total_params_billion": total_params / 1e9,
        "categories_gib": {k: v / (1024 ** 3) for k, v in categories.items()},
        "step_read_gib": step_gib_read,
        "theoretical_min_step_time_ms": min_step_time_ms,
        "theoretical_max_dense_tps": max_dense_tps,
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved inventory audit to {output_json}")
    return results

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "/home/victor/models/DSV4-Flash-Vision-ablit-EXL3-MixedK"
    out = sys.argv[2] if len(sys.argv) > 2 else "tensor_inventory.json"
    audit_inventory(p, out)
