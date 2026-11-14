import unittest

import torch

from sglang_exl3.sidecar import route_exl3_sidecar, split_fused_group, split_sidecar_name
from sglang_exl3.exl3 import storage_kind_for_prefix


class _Owner:
    def __init__(self, sizes):
        self._exl3_lin_output_partition_sizes = list(sizes)


class _Param:
    def __init__(self, key, owner, sink):
        self.key = key
        self._exl3_owner = owner
        self._sink = sink

    def weight_loader(self, param, loaded, shard_id=None):
        self._sink.setdefault(self.key, []).append((shard_id, loaded.clone()))


def _module_params(base, n, suffixes, sink, sizes):
    owner = _Owner(sizes)
    params = {}
    for i in range(n):
        for s in suffixes:
            key = f"{base}.exl3_{i}_{s}"
            params[key] = _Param(key, owner, sink)
    return params


_ALL = ("trellis", "suh", "svh", "mcg", "mul1")


class TestSidecarRouting(unittest.TestCase):
    def test_split_sidecar_name(self):
        self.assertEqual(split_sidecar_name("a.b.trellis"), ("a.b", "trellis"))
        self.assertIsNone(split_sidecar_name("a.b.weight"))

    def test_fused_qkv_split_along_output_tiles(self):
        # out = 64 + 32 + 32 = 128 -> 8 out tiles; in = 64 -> 4 in tiles; k_words 48
        sizes = [64, 32, 32]
        base = "model.layers.0.self_attn.qkv_proj"
        sink = {}
        params = _module_params(base, 3, _ALL, sink, sizes)
        trellis = torch.arange(4 * 8 * 48, dtype=torch.int32).reshape(4, 8, 48).to(torch.int16)
        suh = torch.ones(64, dtype=torch.float16)
        svh = torch.arange(128, dtype=torch.float16)
        mul1 = torch.tensor(1, dtype=torch.int32)
        for name, t in ((base + ".trellis", trellis), (base + ".suh", suh), (base + ".svh", svh), (base + ".mul1", mul1)):
            self.assertTrue(route_exl3_sidecar(params, name, t), name)

        self.assertTrue(torch.equal(sink[f"{base}.exl3_0_trellis"][0][1], trellis[:, 0:4, :]))
        self.assertTrue(torch.equal(sink[f"{base}.exl3_1_trellis"][0][1], trellis[:, 4:6, :]))
        self.assertTrue(torch.equal(sink[f"{base}.exl3_2_trellis"][0][1], trellis[:, 6:8, :]))
        self.assertTrue(torch.equal(sink[f"{base}.exl3_1_svh"][0][1], svh[64:96]))
        self.assertTrue(torch.equal(sink[f"{base}.exl3_2_svh"][0][1], svh[96:128]))
        for i in range(3):
            self.assertTrue(torch.equal(sink[f"{base}.exl3_{i}_suh"][0][1], suh))
            self.assertEqual(sink[f"{base}.exl3_{i}_trellis"][0][0], i)
        self.assertEqual(sink[f"{base}.exl3_0_mul1"][0][1].item(), 1)

    def test_gate_up_and_fused_qkv_a_shards(self):
        sink = {}
        gu = "model.layers.5.mlp.shared_experts.gate_up_proj"
        params = _module_params(gu, 2, _ALL, sink, [48, 48])
        t = torch.zeros(4, 3, 48, dtype=torch.int16)
        self.assertTrue(route_exl3_sidecar(params, "model.layers.5.mlp.shared_experts.gate_proj.trellis", t))
        self.assertTrue(route_exl3_sidecar(params, "model.layers.5.mlp.shared_experts.up_proj.trellis", t))
        self.assertEqual(sink[f"{gu}.exl3_0_trellis"][0][0], 0)
        self.assertEqual(sink[f"{gu}.exl3_1_trellis"][0][0], 1)

        fq = "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa"
        sink2 = {}
        params2 = _module_params(fq, 2, _ALL, sink2, [512, 64])
        self.assertTrue(route_exl3_sidecar(params2, "model.layers.3.self_attn.q_a_proj.svh", torch.zeros(512, dtype=torch.float16)))
        self.assertTrue(route_exl3_sidecar(params2, "model.layers.3.self_attn.kv_a_proj_with_mqa.svh", torch.zeros(64, dtype=torch.float16)))
        self.assertEqual(sink2[f"{fq}.exl3_0_svh"][0][0], 0)
        self.assertEqual(sink2[f"{fq}.exl3_1_svh"][0][0], 1)

    def test_single_shard_and_skips(self):
        sink = {}
        o = "model.layers.0.self_attn.o_proj"
        params = _module_params(o, 1, _ALL, sink, [64])
        self.assertTrue(route_exl3_sidecar(params, o + ".trellis", torch.zeros(4, 4, 48, dtype=torch.int16)))
        self.assertEqual(sink[f"{o}.exl3_0_trellis"][0][0], 0)
        # expert sidecars go through the expert mapping, not here
        self.assertFalse(route_exl3_sidecar(params, "model.layers.0.mlp.experts.3.gate_proj.trellis", torch.zeros(1)))
        # plain weights and names already present are not sidecars
        self.assertFalse(route_exl3_sidecar(params, o + ".weight", torch.zeros(1)))
        # unknown module: not consumed, so the post-load check can fail closed
        self.assertFalse(route_exl3_sidecar(params, "model.layers.9.self_attn.nope.trellis", torch.zeros(1)))

    def test_split_rejects_bad_sizes(self):
        with self.assertRaises(ValueError):
            split_fused_group([24, 16], "svh", torch.zeros(40))
        with self.assertRaises(ValueError):
            split_fused_group([32, 32], "trellis", torch.zeros(2, 3, 16))

    def test_merged_split_sizes_and_create_weights_split(self):
        import torch.nn as nn
        from sglang_exl3.exl3 import EXL3Config
        from sglang_exl3.dense import EXL3LinearMethod

        def ent(out, in_=4096):
            return {"stored_tensors": {
                f"x.{out}.trellis": {"shape": [in_ // 16, out // 16, 48], "dtype": "torch.int16"},
            }}
        ts = {
            "model.language_model.layers.3.self_attn.q_a_proj": ent(1536),
            "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa": ent(512),
            "model.language_model.layers.5.mlp.shared_experts.gate_proj": ent(1024),
            "model.language_model.layers.5.mlp.shared_experts.up_proj": ent(1024),
        }
        cfg = EXL3Config.from_config({"bits": 4, "codebook": "mcg", "tensor_storage": ts})
        self.assertEqual(cfg.merged_split_sizes("model.layers.3.self_attn.fused_qkv_a_proj_with_mqa"), [1536, 512])
        self.assertEqual(cfg.merged_split_sizes("model.layers.5.mlp.shared_experts.gate_up_proj"), [1024, 1024])
        self.assertIsNone(cfg.merged_split_sizes("model.layers.4.self_attn.o_proj"))

        method = EXL3LinearMethod(cfg, bits=4)
        method.split_sizes = cfg.merged_split_sizes("model.layers.3.self_attn.fused_qkv_a_proj_with_mqa")
        layer = nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=4096,
            output_partition_sizes=[2048],
            input_size=4096,
            output_size=2048,
            params_dtype=torch.float16,
        )
        self.assertEqual(len(layer._exl3_lin_packs), 2)
        self.assertEqual(layer._exl3_lin_output_partition_sizes, [1536, 512])
        self.assertTrue(hasattr(layer, "exl3_1_trellis"))
        # a non-matching output size is left alone
        layer2 = nn.Module()
        m2 = EXL3LinearMethod(cfg, bits=4)
        m2.split_sizes = [1536, 512]
        m2.create_weights(layer2, 4096, [1024], 4096, 1024, torch.float16)
        self.assertEqual(len(layer2._exl3_lin_packs), 1)

    def test_storage_kind_resolves_merged_modules(self):
        kinds = {
            "model.layers.5.mlp.shared_experts.gate_proj": "exl3",
            "model.layers.5.mlp.shared_experts.up_proj": "exl3",
            "model.layers.3.self_attn.q_a_proj": "exl3",
            "model.layers.3.self_attn.kv_a_proj_with_mqa": "exl3",
            "model.layers.3.self_attn.indexer.wk": "native",
            "model.layers.9.self_attn.f_a_proj": "native",
        }
        self.assertEqual(storage_kind_for_prefix(kinds, "model.layers.5.mlp.shared_experts.gate_up_proj"), "exl3")
        self.assertEqual(storage_kind_for_prefix(kinds, "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa"), "exl3")
        self.assertEqual(storage_kind_for_prefix(kinds, "model.layers.3.self_attn.indexer.wk"), "native")
        self.assertIsNone(storage_kind_for_prefix(kinds, "model.layers.5.mlp.gate_up_proj"))
        self.assertEqual(storage_kind_for_prefix(kinds, "model.layers.5.self_attn.o_proj"), None)
        with self.assertRaises(RuntimeError):
            storage_kind_for_prefix({"x.gate_proj": "exl3", "x.up_proj": "native"}, "x.gate_up_proj")


if __name__ == "__main__":
    unittest.main()
