#!/usr/bin/env python3
"""Diagnostic to inspect DSV4 checkpoint dtypes, scales, and test SGLang FP8 delegation on GB10."""

import json
import os
import sys
from pathlib import Path
import torch

MODEL_PATH = Path("/home/victor/models/DSV4-Flash-Vision-ablit-EXL3-MixedK")

def inspect_configs():
    print("\n=== 1. Checkpoint Configurations ===")
    cfg_file = MODEL_PATH / "config.json"
    qcfg_file = MODEL_PATH / "quantization_config.json"
    
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            cfg = json.load(f)
        print(f"[config.json] model_type: {cfg.get('model_type')}, torch_dtype: {cfg.get('torch_dtype')}")
        if "quantization_config" in cfg:
            print(f"[config.json] inline quant_config: {json.dumps(cfg['quantization_config'], indent=2)}")
    
    if qcfg_file.exists():
        with open(qcfg_file, "r") as f:
            qcfg = json.load(f)
        print(f"[quantization_config.json]:\n{json.dumps(qcfg, indent=2)}")

def inspect_safetensors():
    print("\n=== 2. Inspecting Safetensors Shard Tensors & Dtypes ===")
    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors not installed, skipping shard inspection")
        return
    
    shard_files = sorted(MODEL_PATH.glob("*.safetensors"))
    if not shard_files:
        print("No safetensors found in", MODEL_PATH)
        return
    
    print(f"Found {len(shard_files)} safetensors shards.")
    
    # Check first shard and search for non-routed layers
    sample_tensors = {}
    for shard in shard_files[:3]:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                # Look for typical non-routed and routed weights
                if any(x in key for x in ["o_proj", "q_b_proj", "kv_b_proj", "shared_expert", "lm_head", "gate_up_proj", "scale"]):
                    if len(sample_tensors) < 20:
                        slice_obj = f.get_slice(key)
                        sample_tensors[key] = {
                            "shape": list(slice_obj.get_shape()),
                            "dtype": str(slice_obj.get_dtype()),
                            "shard": shard.name
                        }
    
    for k, v in sample_tensors.items():
        print(f"  {k}: shape={v['shape']}, dtype={v['dtype']} (in {v['shard']})")

def test_fp8_quant_method():
    print("\n=== 3. Testing SGLang FP8 Quantization Delegation ===")
    try:
        from sglang.srt.layers.quantization import get_quantization_config
        from sglang_exl3.exl3 import EXL3Config
        
        # Test EXL3Config with non_routed_quantization
        qcfg_file = MODEL_PATH / "quantization_config.json"
        if qcfg_file.exists():
            with open(qcfg_file, "r") as f:
                qcfg = json.load(f)
        else:
            qcfg = {
                "quant_method": "exl3",
                "bits": 2,
                "non_routed_quantization": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "weight_block_size": [128, 128]
                }
            }
        
        exl3_cfg = EXL3Config.from_config(qcfg)
        print(f"exl3_cfg.non_routed_dtype_policy: {exl3_cfg.non_routed_dtype_policy}")
        print(f"exl3_cfg.non_routed_quantization: {exl3_cfg.non_routed_quantization}")
        
        delegate = exl3_cfg._non_routed_delegate()
        print(f"EXL3Config._non_routed_delegate(): {delegate}")
        
        # Test with env var SGLANG_EXL3_NON_ROUTED_FP8=1
        os.environ["SGLANG_EXL3_NON_ROUTED_FP8"] = "1"
        delegate = exl3_cfg._non_routed_delegate()
        print(f"EXL3Config._non_routed_delegate() with SGLANG_EXL3_NON_ROUTED_FP8=1: {delegate}")
        
        if delegate is not None:
            print(f"Delegate class: {delegate.__class__.__name__}")
            # Mock a linear layer
            class MockLinear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_features = 4096
                    self.out_features = 4096
            
            mock_layer = MockLinear()
            test_prefixes = [
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.self_attn.q_b_proj",
                "model.layers.0.self_attn.kv_b_proj",
                "model.layers.0.mlp.shared_experts.gate_up_proj",
                "model.layers.0.mlp.shared_experts.down_proj",
                "lm_head",
            ]
            for pfx in test_prefixes:
                qm = exl3_cfg.get_quant_method(mock_layer, pfx)
                print(f"  get_quant_method(MockLinear, '{pfx}'): {qm.__class__.__name__ if qm else None}")
                
    except Exception as e:
        import traceback
        print(f"Exception during FP8 quant test: {e}")
        traceback.print_exc()

def test_fp8_gemm_on_device():
    print("\n=== 4. Testing Hardware FP8 Support on Device ===")
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}, Capability: {torch.cuda.get_device_capability(0)}")
    try:
        a = torch.randn(1, 4096, dtype=torch.bfloat16, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev).to(torch.float8_e4m3fn)
        print("Allocated FP8 tensors on GPU successfully!")
        
        # Test scaled mm if available
        if hasattr(torch, "_scaled_mm"):
            scale_a = torch.tensor(1.0, device=dev)
            scale_b = torch.tensor(1.0, device=dev)
            out = torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
            print("torch._scaled_mm SUCCESS! Output shape:", out.shape)
        else:
            print("torch._scaled_mm not present on this torch build")
    except Exception as e:
        print(f"FP8 GPU test exception: {e}")

if __name__ == "__main__":
    inspect_configs()
    inspect_safetensors()
    test_fp8_quant_method()
    test_fp8_gemm_on_device()
