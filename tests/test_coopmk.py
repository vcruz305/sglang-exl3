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
    COOPMK_MAX_ROWS,
    COOPMK_PLAN,
    _coopmk_act_limit,
    _coopmk_select,
    _exl3_moe_tail_args,
    apply_exl3_experts,
    _exl3_moe_tail_args,
    apply_exl3_experts,
    apply_exl3_fused_moe,
    build_exl3_fused_state,
    coopmk_unsupported_reason,
)

# Positional layout of ext.CoopMK(...) as bound in exllamav3_ext/bindings.cpp (1.6.1): 33 args.
CTOR_ARGS = 33
IDX_HI, IDX_GATE_TRELLIS, IDX_GATE_K, IDX_MCG, IDX_MUL1, IDX_ACT, IDX_LIMIT, IDX_GATED = 0, 1, 13, 16, 17, 18, 19, 20
IDX_CTR, IDX_OUT, IDX_SH, IDX_MIN, IDX_MAX, IDX_PLAN = 27, 28, 29, 30, 31, 32
K_GATE_ARG, K_UP_ARG, K_DOWN_ARG = 10, 11, 12  # plugin exl3_moe positional K args


class FakeCoopMK:
    instances = []
    run_calls = []

    def __init__(self, *args):
        assert len(args) == CTOR_ARGS, len(args)
        self.args = args
        FakeCoopMK.instances.append(self)

    def run(self, x, sel, rw, sh_out=None, wide_a=-1, wide_b=-1, plan=-1):
        FakeCoopMK.run_calls.append((x, sel, rw, sh_out, wide_a, wide_b, plan))
        out = self.args[IDX_OUT]
        bsz = x.shape[0]
        # Deterministic stand-in for the kernel: row r gets r + 1.0 in every column.
        out[:bsz] = torch.arange(1, bsz + 1, dtype=torch.float32).reshape(bsz, 1).expand_as(out[:bsz])


def _fake_ext(with_coopmk=True):
    mod = types.ModuleType("exllamav3_ext")
    mod.calls = []
    mod.exl3_moe_max_concurrency = lambda idx: 1

    def exl3_moe(*args):
        mod.calls.append(args)

    mod.exl3_moe = exl3_moe
    if with_coopmk:
        mod.CoopMK = FakeCoopMK
    return mod


def _linear(k, in_f=128, out_f=128):
    trellis = torch.zeros(in_f // 16, out_f // 16, 16 * k, dtype=torch.int16)
    suh = torch.ones(in_f, dtype=torch.float16)
    svh = torch.ones(out_f, dtype=torch.float16)
    mul1 = torch.tensor([MUL1_MARKER_SIGNED_INT32], dtype=torch.int32)
    return EXL3DenseLinear(trellis, suh, svh, None, mul1)


def _layer(hidden=128, inter=128):
    layer = nn.Module()
    layer.w13_trellis = torch.zeros(1)  # only used for .device
    layer._exl3_bits = 2.32
    layer._exl3_hidden_size = hidden
    layer._exl3_intermediate_local = inter
    layer._exl3_use_mcg = False
    layer._exl3_use_mul1 = True
    return layer


def _inners(ks=(2, 2), hidden=128, inter=128, down_k=None):
    # ks: K per expert for gate and up; down_k: K for every down projection (default: same as gate).
    # The plugin needs one K per projection across experts (_set_fused_k), so gate/up/down are
    # uniform within a projection and may differ between projections.
    out = []
    for k in ks:
        out.append({
            "gate": _linear(k, hidden, inter),
            "up": _linear(k, hidden, inter),
            "down": _linear(down_k or k, inter, hidden),
        })
    return out


ENV = {"SGLANG_EXL3_MOE_KERNEL": "exllamav3", "SGLANG_EXL3_MOE_COOPMK": "1"}


class TestCoopMKHelpers(unittest.TestCase):
    def test_act_limit_mapping(self):
        self.assertEqual(_coopmk_act_limit(float("inf")), 0.0)  # infinite limit: no clamp
        self.assertIsNone(_coopmk_act_limit(0.0))  # 0 is the kernel's no-clamp sentinel, not ours
        self.assertIsNone(_coopmk_act_limit(float("nan")))
        self.assertEqual(_coopmk_act_limit(7.5), 7.5)

    def test_reason_flag_off(self):
        layer = _layer()
        with mock.patch.dict(os.environ, {"SGLANG_EXL3_MOE_COOPMK": "0"}):
            self.assertIn("not set", coopmk_unsupported_reason(layer, _inners(), float("inf")))


class TestCoopMKState(unittest.TestCase):
    def setUp(self):
        FakeCoopMK.instances = []
        FakeCoopMK.run_calls = []

    def _apply(self, ext, *args):
        with mock.patch.dict(sys.modules, {"exllamav3_ext": ext}), mock.patch.dict(os.environ, ENV):
            return apply_exl3_fused_moe(*args)

    def _build(self, layer=None, inners=None, ext=None, env=ENV):
        layer = layer or _layer()
        inners = inners or _inners()
        ext = ext or _fake_ext()
        with mock.patch.dict(sys.modules, {"exllamav3_ext": ext}), mock.patch.dict(os.environ, env):
            build_exl3_fused_state(layer, inners)
        return layer, inners, ext

    def test_ctor_arguments_follow_native_setup(self):
        layer, inners, _ = self._build(inners=_inners(ks=(2, 2), down_k=3))
        self.assertEqual(len(FakeCoopMK.instances), 1)
        a = FakeCoopMK.instances[0].args
        self.assertEqual(a[IDX_HI], 128)
        ptrs = layer._exl3_ptrs
        self.assertIs(a[IDX_GATE_TRELLIS], ptrs["gate_trellis"])
        self.assertIs(a[IDX_GATE_TRELLIS + 8], ptrs["down_svh"])  # last of nine pointer tables
        self.assertIsNone(a[10]); self.assertIsNone(a[11]); self.assertIsNone(a[12])  # no biases
        self.assertEqual(a[IDX_GATE_K].tolist(), [2, 2])
        self.assertEqual(a[IDX_GATE_K + 1].tolist(), [2, 2])
        self.assertEqual(a[IDX_GATE_K + 2].tolist(), [3, 3])
        self.assertEqual(a[IDX_GATE_K].dtype, torch.int32)
        self.assertFalse(a[IDX_MCG])
        self.assertTrue(a[IDX_MUL1])
        self.assertEqual(a[IDX_ACT], 0)
        self.assertEqual(a[IDX_LIMIT], 0.0)  # SWIGLU_LIMIT_DEFAULT is inf -> no clamp
        self.assertTrue(a[IDX_GATED])
        self.assertIsNone(a[IDX_SH])
        self.assertEqual((a[IDX_MIN], a[IDX_MAX]), (-1, -1))
        self.assertEqual(a[IDX_PLAN], COOPMK_PLAN)
        slots = COOPMK_MAX_ROWS * 8
        self.assertEqual(tuple(a[IDX_CTR].shape), (slots * (128 // 128) + COOPMK_MAX_ROWS * (128 // 128) + 2 + (slots + 1) + slots,))
        self.assertEqual(a[IDX_CTR].dtype, torch.int32)
        self.assertEqual(tuple(a[IDX_OUT].shape), (COOPMK_MAX_ROWS, 128))
        self.assertEqual(a[IDX_OUT].dtype, torch.float32)
        self.assertIsNone(layer._exl3_coopmk_reason)

    def test_small_batch_dispatches_to_coopmk_not_exl3_moe(self):
        layer, inners, ext = self._build()
        x = torch.randn(4, 128)
        ids = torch.tensor([[0, 1]] * 4, dtype=torch.long)
        weights = torch.full((4, 2), 1.25)  # routed scale already applied upstream (2.5 * 0.5)
        out = self._apply(ext, x, ids, weights, layer, inners, None, float("inf"))
        self.assertEqual(ext.calls, [], "exl3_moe must not run when CoopMK takes the call")
        self.assertEqual(len(FakeCoopMK.run_calls), 1)
        xh, sel, rw, sh, wa, wb, plan = FakeCoopMK.run_calls[0]
        self.assertEqual(xh.dtype, torch.float16)
        self.assertTrue(torch.equal(xh, x.half()))
        self.assertEqual(sel.dtype, torch.long)
        self.assertTrue(torch.equal(sel, ids))
        self.assertEqual(rw.dtype, torch.float16)
        self.assertTrue(torch.equal(rw, weights.half()))  # passed unchanged, no extra scale
        self.assertIsNone(sh)
        self.assertEqual((wa, wb, plan), (-1, -1, -1))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (4, 128))
        self.assertTrue(torch.equal(out, torch.arange(1, 5, dtype=torch.float32).reshape(4, 1).expand(4, 128)))
        self.assertEqual(layer._exl3_last_apply, "coopmk")

    def test_output_is_a_copy_not_the_shared_buffer(self):
        layer, inners, _ = self._build()
        x = torch.randn(2, 128)
        ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        weights = torch.full((2, 2), 0.5)
        out = self._apply(_fake_ext(), x, ids, weights, layer, inners, None, float("inf"))
        shared = FakeCoopMK.instances[0].args[IDX_OUT]
        shared.fill_(-7.0)  # the next layer's launch would overwrite the shared buffer
        self.assertTrue(torch.equal(out, torch.arange(1, 3, dtype=torch.float32).reshape(2, 1).expand(2, 128)))

    def test_nine_rows_fall_back_to_exl3_moe(self):
        layer, inners, ext = self._build()
        x = torch.randn(9, 128)
        ids = torch.tensor([[0, 1]] * 9, dtype=torch.long)
        weights = torch.full((9, 2), 0.5)
        self._apply(ext, x, ids, weights, layer, inners, None, float("inf"))
        self.assertEqual(FakeCoopMK.run_calls, [])
        self.assertEqual(len(ext.calls), 1)
        self.assertEqual(layer._exl3_last_apply, "fused")

    def test_flag_off_builds_nothing(self):
        layer, inners, ext = self._build(env={"SGLANG_EXL3_MOE_KERNEL": "exllamav3", "SGLANG_EXL3_MOE_COOPMK": "0"})
        self.assertEqual(FakeCoopMK.instances, [])
        x = torch.randn(2, 128)
        ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with mock.patch.dict(sys.modules, {"exllamav3_ext": ext}), mock.patch.dict(os.environ, {"SGLANG_EXL3_MOE_KERNEL": "exllamav3", "SGLANG_EXL3_MOE_COOPMK": "0"}):
            apply_exl3_fused_moe(x, ids, torch.full((2, 2), 0.5), layer, inners, None, float("inf"))
        self.assertEqual(len(ext.calls), 1)
        self.assertEqual(FakeCoopMK.run_calls, [])

    def test_mcg_codebook_falls_back(self):
        layer = _layer()
        layer._exl3_use_mcg = True
        layer._exl3_use_mul1 = False
        _, _, _ = self._build(layer=layer)
        self.assertEqual(FakeCoopMK.instances, [])
        self.assertIn("mul1", layer._exl3_coopmk_reason or "")

    def test_non_integer_k_falls_back(self):
        inners = _inners()
        inners[0]["up"].K = 2.5
        layer, _, _ = self._build(inners=inners)
        self.assertEqual(FakeCoopMK.instances, [])
        self.assertIn("K=2.5", layer._exl3_coopmk_reason)

    def test_shape_outside_128_falls_back(self):
        layer = _layer(hidden=128, inter=64)
        inners = _inners(hidden=128, inter=64)
        self._build(layer=layer, inners=inners)
        self.assertEqual(FakeCoopMK.instances, [])
        self.assertIn("not CoopMK-compatible", layer._exl3_coopmk_reason)

    def test_expert_map_falls_back(self):
        layer, inners, ext = self._build()
        emap = torch.arange(2, dtype=torch.long)
        x = torch.randn(2, 128)
        ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self._apply(ext, x, ids, torch.full((2, 2), 0.5), layer, inners, emap, float("inf"))
        self.assertEqual(FakeCoopMK.run_calls, [])
        self.assertEqual(len(ext.calls), 1)

    def test_extension_without_coopmk_falls_back(self):
        layer, inners, ext = self._build(ext=_fake_ext(with_coopmk=False))
        self.assertEqual(FakeCoopMK.instances, [])
        self.assertIn("no CoopMK", layer._exl3_coopmk_reason)

    def test_finite_limit_builds_its_own_object_and_zero_limit_falls_back(self):
        layer, inners, ext = self._build()
        x = torch.randn(2, 128)
        ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        w = torch.full((2, 2), 0.5)
        self._apply(ext, x, ids, w, layer, inners, None, 7.0)
        self.assertEqual(len(FakeCoopMK.instances), 2)
        self.assertEqual(FakeCoopMK.instances[1].args[IDX_LIMIT], 7.0)
        self._apply(ext, x, ids, w, layer, inners, None, 7.0)  # cached: no third object
        self.assertEqual(len(FakeCoopMK.instances), 2)
        self.assertEqual(len(FakeCoopMK.run_calls), 2)
        self._apply(ext, x, ids, w, layer, inners, None, 0.0)  # 0 would mean no clamp in the kernel
        self.assertEqual(len(FakeCoopMK.run_calls), 2)
        self.assertEqual(len(ext.calls), 1)

    def test_non_long_ids_fall_back(self):
        layer, inners, ext = self._build()
        ids32 = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        with mock.patch.dict(os.environ, ENV):
            self.assertIsNone(_coopmk_select(layer, inners, float("inf"), None, 2, 2, ids32))
            self.assertIsNotNone(_coopmk_select(layer, inners, float("inf"), None, 2, 2, ids32.long()))
        self.assertEqual(FakeCoopMK.run_calls, [])

    def test_apply_exl3_experts_dispatches_with_fused_moe_off(self):
        # Campaign launch sets EXL3_FUSED_MOE=0: apply_exl3_fused_moe is never reached, so the
        # dispatch must happen in apply_exl3_experts (the real forward entry point).
        layer, inners, ext = self._build()
        layer._exl3_inners = inners
        x = torch.randn(4, 128)
        ids = torch.tensor([[0, 1]] * 4, dtype=torch.long)
        weights = torch.full((4, 2), 1.25)
        env = dict(ENV, EXL3_FUSED_MOE="0")
        with mock.patch.dict(sys.modules, {"exllamav3_ext": ext}), mock.patch.dict(os.environ, env):
            out = apply_exl3_experts(x, ids, weights, layer, limit=float("inf"))
        self.assertEqual(len(FakeCoopMK.run_calls), 1)
        self.assertEqual(ext.calls, [])
        self.assertEqual(layer._exl3_last_apply, "coopmk")
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(tuple(out.shape), (4, 128))

    def test_exl3_moe_tail_args_follow_binding_arity(self):
        ext = _fake_ext()
        self.assertEqual(_exl3_moe_tail_args(ext.exl3_moe), ())  # 1.4.4-style binding: 30 args
        ext.exl3_moe.__doc__ = "exl3_moe(arg0: Tensor, ..., arg29: int, arg30: Tensor | None, arg34: int) -> None"
        self.assertEqual(_exl3_moe_tail_args(ext.exl3_moe), (None, None, 0, 1 << 30, 16))

    def test_fused_exl3_moe_call_gets_1_6_1_tail_args(self):
        layer, inners, ext = self._build()
        ext.exl3_moe.__doc__ = "exl3_moe(arg0: int, arg1: int, arg2: int, arg3: int, arg4: int, arg5: int, arg6: int, arg7: int, arg8: int, arg9: int, arg10: int, arg11: int, arg12: int, arg13: int, arg14: int, arg15: int, arg16: int, arg17: int, arg18: int, arg19: int, arg20: int, arg21: int, arg22: int, arg23: int, arg24: int, arg25: int, arg26: int, arg27: int, arg28: int, arg29: int, arg30: int, arg31: int, arg32: int, arg33: int, arg34: int) -> None"
        x = torch.randn(9, 128)  # > COOPMK_MAX_ROWS: stays on the exl3_moe launch
        ids = torch.tensor([[0, 1]] * 9, dtype=torch.long)
        self._apply(ext, x, ids, torch.full((9, 2), 0.5), layer, inners, None, float("inf"))
        self.assertEqual(len(ext.calls), 1)
        self.assertEqual(len(ext.calls[0]), 35)
        self.assertEqual(ext.calls[0][30:], (None, None, 0, 1 << 30, 16))


if __name__ == "__main__":
    unittest.main()
