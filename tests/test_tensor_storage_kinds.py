import unittest

from sglang_exl3.exl3 import EXL3Config, _tensor_storage_kinds


class TestTensorStorageKinds(unittest.TestCase):
    def test_exl3_native_and_empty_entries(self):
        ts = {
            "model.language_model.layers.3.self_attn.q_a_proj": {
                "stored_tensors": {
                    "model.language_model.layers.3.self_attn.q_a_proj.trellis": {},
                    "model.language_model.layers.3.self_attn.q_a_proj.suh": {},
                }
            },
            "model.language_model.layers.3.mlp.gate": {
                "stored_tensors": {
                    "model.language_model.layers.3.mlp.gate.weight": {},
                    "model.language_model.layers.3.mlp.gate.e_score_correction_bias": {},
                }
            },
            "hc_expand": {"stored_tensors": {}},
        }
        kinds = _tensor_storage_kinds(ts)
        self.assertEqual(kinds["model.language_model.layers.3.self_attn.q_a_proj"], "exl3")
        self.assertEqual(kinds["model.language_model.layers.3.mlp.gate"], "native")
        self.assertNotIn("hc_expand", kinds)
        self.assertEqual(_tensor_storage_kinds(None), {})

    def test_language_model_prefix_is_indexed_as_sglang_prefix(self):
        ts = {
            "model.language_model.layers.3.self_attn.q_b_proj": {
                "stored_tensors": {"model.language_model.layers.3.self_attn.q_b_proj.trellis": {}}
            },
            "model.language_model.layers.3.self_attn.indexer.weights_proj": {
                "stored_tensors": {"model.language_model.layers.3.self_attn.indexer.weights_proj.weight": {}}
            },
        }
        kinds = _tensor_storage_kinds(ts)
        self.assertEqual(kinds["model.layers.3.self_attn.q_b_proj"], "exl3")
        self.assertEqual(kinds["model.layers.3.self_attn.indexer.weights_proj"], "native")
        self.assertEqual(kinds["model.language_model.layers.3.self_attn.q_b_proj"], "exl3")

    def test_bits_from_trellis_shape_per_module_and_per_expert_layer(self):
        from sglang_exl3.exl3 import _tensor_storage_bits

        def tr(shape):
            return {"stored_tensors": {"x.trellis": {"shape": shape}}}

        ts = {
            "model.language_model.layers.3.self_attn.q_b_proj": tr([64, 128, 48]),
            "model.language_model.layers.3.self_attn.o_proj": tr([64, 128, 64]),
            "model.language_model.layers.3.mlp.experts.0.up_proj": tr([256, 128, 48]),
            "model.language_model.layers.3.mlp.experts.7.down_proj": tr([128, 256, 48]),
            "model.language_model.layers.6.mlp.experts.0.up_proj": tr([256, 128, 32]),
        }
        mod, lay = _tensor_storage_bits(ts)
        self.assertEqual(mod["model.layers.3.self_attn.q_b_proj"], 3)
        self.assertEqual(mod["model.layers.3.self_attn.o_proj"], 4)
        self.assertEqual(lay, {3: 3, 6: 2})

    def test_mixed_expert_bits_in_one_layer_fail_closed(self):
        from sglang_exl3.exl3 import _tensor_storage_bits

        ts = {
            "model.language_model.layers.3.mlp.experts.0.up_proj": {"stored_tensors": {"a.trellis": {"shape": [1, 1, 48]}}},
            "model.language_model.layers.3.mlp.experts.1.up_proj": {"stored_tensors": {"b.trellis": {"shape": [1, 1, 32]}}},
        }
        with self.assertRaises(ValueError):
            _tensor_storage_bits(ts)

    def test_bits_for_prefix_prefers_storage(self):
        cfg = EXL3Config.from_config({
            "quant_method": "exl3",
            "bits": 2,
            "codebook": "mul1",
            "tensor_storage": {
                "model.language_model.layers.3.mlp.experts.0.up_proj": {
                    "stored_tensors": {"model.language_model.layers.3.mlp.experts.0.up_proj.trellis": {"shape": [256, 128, 48]}}
                },
                "model.language_model.layers.3.self_attn.o_proj": {
                    "stored_tensors": {"model.language_model.layers.3.self_attn.o_proj.trellis": {"shape": [64, 128, 64]}}
                },
            },
        })
        self.assertEqual(cfg.bits_for_prefix("model.layers.3.mlp.experts"), 3)
        self.assertEqual(cfg.bits_for_prefix("model.layers.3.self_attn.o_proj"), 4)
        self.assertEqual(cfg.bits_for_prefix("model.layers.9.self_attn.q_a_proj"), 2)

    def test_from_config_carries_kinds_and_default_is_empty(self):
        cfg = EXL3Config.from_config({
            "quant_method": "exl3",
            "bits": 2,
            "codebook": "mul1",
            "tensor_storage": {
                "m.b_proj": {"stored_tensors": {"m.b_proj.weight": {}}},
                "m.qkv_proj": {"stored_tensors": {"m.qkv_proj.trellis": {}}},
            },
        })
        self.assertEqual(cfg.tensor_storage_kinds, {"m.b_proj": "native", "m.qkv_proj": "exl3"})
        self.assertEqual(EXL3Config(bits=4, codebook="mcg").tensor_storage_kinds, {})


if __name__ == "__main__":
    unittest.main()
