#!/usr/bin/env python3
"""In-Process SGLang Engine Benchmark for NVIDIA DGX Spark GB10 sm_121.

Executes generation directly through sglang.Engine without HTTP, FastAPI,
or ZeroMQ socket IPC overhead to measure true Blackwell engine decode speed.
"""

import os
import sys
import time
import torch

def benchmark_engine(
    model_path: str = "/home/victor/models/DSV4-Flash-Vision-ablit-EXL3-MixedK",
    mode: str = "dspark", # "dense", "eagle", "dspark"
    num_tokens: int = 128,
    disable_radix_cache: bool = False,
    draft_tokens: int = 4,
    kv_cache_dtype: str = "fp8",
    dspark_block_size: int = None,
):
    # DeepSeek-V4's SGLang hook uses the explicit FP8 format name
    # ``fp8_e4m3``.  Accept the vLLM-style ``fp8`` spelling in the benchmark
    # CLI and normalize it before constructing ServerArgs.
    if kv_cache_dtype == "fp8":
        kv_cache_dtype = "fp8_e4m3"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["SGLANG_EXL3_MOE_KERNEL"] = "native"
    os.environ["EXL3_FUSED_MOE"] = "1"
    os.environ["SGLANG_QUANT_ALLOW_DOWNCASTING"] = "1"
    os.environ["SGLANG_EXL3_NON_ROUTED_FP8"] = os.environ.get("SGLANG_EXL3_NON_ROUTED_FP8", "1")
    os.environ["SGLANG_EXL3_MTP_FP8"] = os.environ.get("SGLANG_EXL3_MTP_FP8", "1")
    os.environ["SGLANG_EXL3_NATIVE_MOE_MAX_M"] = os.environ.get("SGLANG_EXL3_NATIVE_MOE_MAX_M", "16")
    os.environ["SGLANG_SM120_FLASHMLA_BACKEND"] = os.environ.get("SGLANG_SM120_FLASHMLA_BACKEND", "flashinfer")
    os.environ["SGLANG_ENABLE_SPEC_V2"] = os.environ.get("SGLANG_ENABLE_SPEC_V2", "1")

    from sglang import Engine

    mem_frac = float(os.environ.get("MEM_FRACTION_STATIC", "0.70"))

    engine_kwargs = {
        "model_path": model_path,
        "quantization": "exl3",
        "load_format": "exl3_mmap",
        "mem_fraction_static": mem_frac,
        "max_running_requests": 1,
        "chunked_prefill_size": 1024,
        "kv_cache_dtype": kv_cache_dtype,
        "disable_radix_cache": disable_radix_cache,
    }

    if mode == "eagle":
        engine_kwargs.update({
            "speculative_algorithm": "EAGLE",
            "speculative_draft_model_path": model_path,
            "speculative_num_steps": draft_tokens,
            "speculative_eagle_topk": 1,
        })
    elif mode == "dspark":
        resolved_block_size = dspark_block_size if dspark_block_size is not None else (draft_tokens - 1)
        engine_kwargs.update({
            "speculative_algorithm": "DSPARK",
            "speculative_draft_model_path": model_path,
            "speculative_num_steps": 1,
            "speculative_num_draft_tokens": draft_tokens,
            "speculative_dspark_block_size": resolved_block_size,
        })
        print(f"DSpark Configuration: gamma (proposals) = {resolved_block_size}, verify window = {draft_tokens}")

    print(f"\n=======================================================")
    print(f"Initializing in-process SGLang Engine (Mode: {mode.upper()}, Tokens: {num_tokens}, DisableRadix: {disable_radix_cache})...")
    print(f"=======================================================")

    engine = Engine(**engine_kwargs)

    prompt = "Explain quantum computing and its potential advantages over classical computing in two detailed paragraphs."

    # Warmup
    print("\nRunning warmup generation...")
    _ = engine.generate(prompt=prompt, sampling_params={"max_new_tokens": 16, "temperature": 0.0})

    # Benchmark sustained generation
    print(f"Benchmarking sustained generation ({num_tokens} tokens)...")
    t0 = time.perf_counter()
    res = engine.generate(prompt=prompt, sampling_params={"max_new_tokens": num_tokens, "temperature": 0.0})
    elapsed = time.perf_counter() - t0

    text = res.get("text", "")
    meta_info = res.get("meta_info", {})
    output_ids = res.get("output_ids", [])
    if isinstance(meta_info, dict) and "completion_tokens" in meta_info and meta_info["completion_tokens"] is not None:
        tokens_gen = int(meta_info["completion_tokens"])
    elif output_ids and len(output_ids) > 0:
        tokens_gen = len(output_ids)
    elif isinstance(meta_info, dict) and "output_tokens" in meta_info and meta_info["output_tokens"] is not None:
        tokens_gen = int(meta_info["output_tokens"])
    else:
        # Require authoritative count; do not invent generated count
        raise RuntimeError(
            f"Authoritative completion token count not found in Engine response! "
            f"meta_info={meta_info}, text_len={len(text)}"
        )
    tok_per_sec = tokens_gen / elapsed if elapsed > 0 else 0.0

    print("\n--- Benchmark Result ---")
    print(f"Mode:                    {mode.upper()}")
    print(f"Disable Radix Cache:     {disable_radix_cache}")
    print(f"Elapsed Time:            {elapsed:.4f} s")
    print(f"Tokens Generated:        {tokens_gen} (authoritative)")
    print(f"Sustained Throughput:    {tok_per_sec:.2f} tok/s")
    print(f"Average Step Latency:    {(elapsed / tokens_gen) * 1000.0:.2f} ms/token")
    print(f"Output Preview:          {text[:150].strip()}...")
    print("------------------------\n")

    engine.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="In-Process SGLang Engine Benchmark")
    parser.add_argument("--mode", choices=["eagle", "dspark", "dense"], default="dspark", help="Generation mode")
    parser.add_argument("--tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--draft-tokens", type=int, default=4, help="Number of speculative draft tokens (or steps for EAGLE)")
    parser.add_argument("--dspark-block-size", type=int, default=None, help="DSpark proposal block size (gamma). Default: draft-tokens - 1")
    parser.add_argument("--kv-cache-dtype", default="fp8", help="KV cache dtype (fp8 or auto)")
    parser.add_argument("--disable-radix-cache", action="store_true", help="Disable prefix radix cache")
    parser.add_argument("--model-path", default="/home/victor/models/DSV4-Flash-Vision-ablit-EXL3-MixedK", help="Model path")
    args, unknown = parser.parse_known_args()

    # Backwards compatibility for positional args: python script.py <mode> <tokens>
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        mode_arg = sys.argv[1]
        tokens_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 128
        disable_radix = "--disable-radix-cache" in sys.argv
        benchmark_engine(
            model_path=args.model_path,
            mode=mode_arg,
            num_tokens=tokens_arg,
            disable_radix_cache=disable_radix,
            draft_tokens=args.draft_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
            dspark_block_size=args.dspark_block_size,
        )
    else:
        benchmark_engine(
            model_path=args.model_path,
            mode=args.mode,
            num_tokens=args.tokens,
            disable_radix_cache=args.disable_radix_cache,
            draft_tokens=args.draft_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
            dspark_block_size=args.dspark_block_size,
        )
