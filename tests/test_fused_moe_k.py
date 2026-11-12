import os
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from sglang_exl3.dense import EXL3DenseLinear
from sglang_exl3.exl3 import (
    MUL1_MARKER_SIGNED_INT32,
    apply_exl3_fused_moe,
    build_exl3_fused_state,
)

# Positional indices of the per-projection K arguments in the exl3_moe call
# (see apply_exl3_fused_moe: args = (xh, out, counts, tok, w, t0..t3, act, k_gate, k_up, k_down, ...)).
K_GATE_ARG, K_UP_ARG, K_DOWN_ARG = 10, 11, 12


def _fake_ext():
    mod = types.ModuleType("exllamav3_ext")
    mod.calls = []
    mod.exl3_moe_max_concurrency = lambda idx: 1

    def exl3_moe(*args):
        mod.calls.append(args)

    mod.exl3_moe = exl3_moe
    return mod


def _linear(k, in_f=32, out_f=32):
    # Trellis last dim is 16*K (I16), first two dims are 16-wide tiles.
    trellis = torch.zeros(in_f // 16, out_f // 16, 16 * k, dtype=torch.int16)
    suh = torch.ones(in_f, dtype=torch.float16)
    svh = torch.ones(out_f, dtype=torch.float16)
    mul1 = torch.tensor([MUL1_MARKER_SIGNED_INT32], dtype=torch.int32)
    return EXL3DenseLinear(trellis, suh, svh, None, mul1)


def _layer(bits=2.32):
    layer = nn.Module()
    layer.w13_trellis = torch.zeros(1)  # only used for .device
    layer._exl3_bits = bits  # layer-wide bits from config: int(2.32) == 2
    layer._exl3_hidden_size = 32
    layer._exl3_intermediate_local = 32
    layer._exl3_use_mcg = False
    layer._exl3_use_mul1 = True
    return layer


class TestFusedMoEPerProjectionK(unittest.TestCase):
    def _build_and_apply(self, inners, layer=None):
        layer = layer or _layer()
        ext = _fake_ext()
        x = torch.zeros(2, 32)
        ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        weights = torch.full((2, 2), 0.5)
        env = {"SGLANG_EXL3_MOE_KERNEL": "exllamav3"}
        with mock.patch.dict(sys.modules, {"exllamav3_ext": ext}), mock.patch.dict(os.environ, env):
            build_exl3_fused_state(layer, inners)
            apply_exl3_fused_moe(x, ids, weights, layer, inners, None, float("inf"))
        self.assertEqual(len(ext.calls), 1, "exl3_moe should be called exactly once")
        return ext.calls[0]

    def test_k3_projections_pass_k3_not_layer_wide_k2(self):
        # Pack reality: routed experts are K=3 (trellis last dim 48) under a 2.32-bpw config.
        inners = [{"gate": _linear(3), "up": _linear(3), "down": _linear(3)} for _ in range(2)]
        args = self._build_and_apply(inners)
        self.assertEqual(
            (args[K_GATE_ARG], args[K_UP_ARG], args[K_DOWN_ARG]),
            (3, 3, 3),
            "fused exl3_moe must take K per projection from the trellis shape, not int(bits)=2",
        )

    def test_uniform_k2_passes_k2(self):
        inners = [{"gate": _linear(2), "up": _linear(2), "down": _linear(2)} for _ in range(2)]
        args = self._build_and_apply(inners)
        self.assertEqual((args[K_GATE_ARG], args[K_UP_ARG], args[K_DOWN_ARG]), (2, 2, 2))

    def test_k_differs_by_projection_is_passed_per_projection(self):
        inners = [{"gate": _linear(3), "up": _linear(3), "down": _linear(2)} for _ in range(2)]
        args = self._build_and_apply(inners)
        self.assertEqual((args[K_GATE_ARG], args[K_UP_ARG], args[K_DOWN_ARG]), (3, 3, 2))

    def test_mixed_k_across_experts_in_one_projection_raises(self):
        inners = [
            {"gate": _linear(2), "up": _linear(3), "down": _linear(3)},
            {"gate": _linear(3), "up": _linear(3), "down": _linear(3)},
        ]
        layer = _layer()
        ext = _fake_ext()
        with mock.patch.dict(sys.modules, {"exllamav3_ext": ext}):
            with self.assertRaisesRegex(RuntimeError, r"gate has K=\[2, 3\]"):
                build_exl3_fused_state(layer, inners)


if __name__ == "__main__":
    unittest.main()
