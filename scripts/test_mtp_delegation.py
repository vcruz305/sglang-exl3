#!/usr/bin/env python3
"""Diagnostic script to verify MTP FP8 delegation on NVIDIA DGX Spark GB10 sm_121."""

import os
import sys
import torch
import torch.nn as nn

def test_mtp_delegation():
    print("============================================================")
    print("Testing MTP FP8 Quantization Delegation")
    print("============================================================")

    # Enable MTP FP8 delegation
    os.environ["SGLANG_EXL3_MTP_FP8"] = "1"
    os.environ["SGLANG_EXL3_NON_ROUTED_FP8"] = "1"

    from sglang_exl3.exl3 import EXL3Config, EXL3MoEMethod, EXL3LinearMethod

    cfg = EXL3Config(bits=2)

    class MockMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_local_experts = 256

    class MockLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = 4096
            self.out_features = 4096

    mock_moe = MockMoE()
    mock_linear = MockLinear()

    test_cases = [
        (mock_moe, "model.layers.0.mlp.experts", "Target MoE", False),
        (mock_linear, "model.layers.0.self_attn.o_proj", "Target Dense Linear", False),
        (mock_moe, "model.layers.43.mlp.experts", "MTP Layer 43 MoE", True),
        (mock_linear, "model.layers.43.self_attn.o_proj", "MTP Layer 43 Linear", True),
        (mock_moe, "model.mtp_layers.0.mlp.experts", "MTP Named MoE", True),
        (mock_linear, "model.mtp_layers.0.self_attn.o_proj", "MTP Named Linear", True),
        (mock_moe, "stages.0.mlp.experts", "DSpark Stage 0 MoE", True),
        (mock_linear, "stages.0.self_attn.o_proj", "DSpark Stage 0 Linear", True),
        (mock_moe, "stages.1.mlp.experts", "DSpark Stage 1 MoE", True),
        (mock_moe, "stages.2.mlp.experts", "DSpark Stage 2 MoE", True),
    ]

    try:
        import sglang
        sglang_available = True
    except ImportError:
        sglang_available = False

    print(f"SGLang available in environment: {sglang_available}")
    print("\nEvaluating quant methods for target vs MTP layers (MTP FP8 = 1):")
    for layer, prefix, desc, is_mtp in test_cases:
        qm = cfg.get_quant_method(layer, prefix)
        qm_name = qm.__class__.__name__ if qm else "None (Unquantized)"
        print(f"  [{desc:22s}] prefix='{prefix}': {qm_name}")

        if not is_mtp:
            if isinstance(layer, MockMoE):
                assert isinstance(qm, EXL3MoEMethod), f"Expected EXL3MoEMethod for target MoE, got {qm_name}"
            elif sglang_available:
                assert "Fp8" in qm_name, f"Expected Fp8LinearMethod for target dense linear, got {qm_name}"
        else:
            assert not isinstance(qm, EXL3MoEMethod), f"MTP MoE unexpectedly got EXL3MoEMethod: {qm_name}"
            if sglang_available:
                assert "Fp8" in qm_name, f"Expected Fp8 method for MTP layer {prefix}, got {qm_name}"

    # Test independent policy: MTP FP8 = 0, Non-Routed FP8 = 1
    # Draft layers should remain unquantized (None / BF16), NOT fall through to EXL3 or FP8
    print("\nEvaluating independent policy (MTP FP8 = 0, Non-Routed FP8 = 1):")
    os.environ["SGLANG_EXL3_MTP_FP8"] = "0"
    os.environ["SGLANG_EXL3_NON_ROUTED_FP8"] = "1"
    cfg_independent = EXL3Config(bits=2)
    for layer, prefix, desc, is_mtp in test_cases:
        qm = cfg_independent.get_quant_method(layer, prefix)
        qm_name = qm.__class__.__name__ if qm else "None (Unquantized BF16)"
        if is_mtp:
            assert qm is None, f"Expected unquantized (None) for draft layer {prefix} when MTP_FP8=0, got {qm_name}"
            assert not isinstance(qm, EXL3MoEMethod), f"Draft layer {prefix} must NEVER be EXL3MoEMethod"
        print(f"  [{desc:22s}] prefix='{prefix}': {qm_name}")

    print("\nAll MTP delegation assertions passed successfully!")

if __name__ == "__main__":
    test_mtp_delegation()
