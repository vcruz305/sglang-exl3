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
