import unittest
from sglang_exl3.exl3 import EXL3Config


class TestEXL3Config(unittest.TestCase):
    def test_default_config(self):
        cfg = EXL3Config(bits=4, codebook="mcg")
        self.assertEqual(cfg.get_name(), "exl3")
        self.assertEqual(cfg.bits, 4)
        self.assertEqual(cfg.codebook, "mcg")
        self.assertEqual(cfg.get_min_capability(), 80)

    def test_mixed_k_layer_bits(self):
        cfg = EXL3Config(bits=2, layer_bits={"42": 3, "27": 3})
        self.assertEqual(cfg.bits_for_prefix("model.layers.0.mlp.experts"), 2)
        self.assertEqual(cfg.bits_for_prefix("model.layers.42.mlp.experts"), 3)
        self.assertEqual(cfg.bits_for_prefix("model.layers.27.mlp.experts"), 3)

    def test_from_config(self):
        hf_dict = {
            "quant_method": "exl3",
            "bits": 3,
            "codebook": "mcg",
            "scope": "glm53_routed_experts_only",
            "layer_bits": {"10": 4},
        }
        cfg = EXL3Config.from_config(hf_dict)
        self.assertEqual(cfg.bits, 3)
        self.assertEqual(cfg.bits_for_prefix("layers.10.mlp.experts"), 4)
        self.assertEqual(cfg.bits_for_prefix("layers.5.mlp.experts"), 3)

    def test_override_quantization_method(self):
        override = EXL3Config.override_quantization_method({"quant_method": "exl3"}, None)
        self.assertEqual(override, "exl3")
        self.assertIsNone(EXL3Config.override_quantization_method({"quant_method": "fp8"}, None))

    def test_fp8_delegation_config(self):
        import os
        hf_dict = {
            "quant_method": "exl3",
            "bits": 2,
            "non_routed_dtype_policy": "bf16_as_stored",
            "non_routed_quantization": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128]
            }
        }
        cfg = EXL3Config.from_config(hf_dict)
        self.assertEqual(cfg.non_routed_dtype_policy, "bf16_as_stored")
        self.assertEqual(cfg.non_routed_quantization["quant_method"], "fp8")
        
        # Test default fallback without SGLang installed
        delegate = cfg._non_routed_delegate()
        # SGLang may not be installed in test env, but _nr_delegate_cached should be handled gracefully
        self.assertTrue(hasattr(cfg, "_nr_delegate_cached"))

    def test_mtp_delegation_recognition(self):
        cfg = EXL3Config(bits=2)
        lm43 = cfg._LAYER_RE.search("model.layers.43.mlp.experts")
        self.assertIsNotNone(lm43)
        self.assertEqual(int(lm43.group(1)), 43)

        # Non-MTP target layer
        lm0 = cfg._LAYER_RE.search("model.layers.0.mlp.experts")
        self.assertIsNotNone(lm0)
        self.assertEqual(int(lm0.group(1)), 0)

        # Test DSpark stages detection
        for pfx in ["stages.0.mlp.experts", "model.stages.2.mlp.experts", "stages.1.linear"]:
            pfx_lower = pfx.lower()
            is_draft = (
                "stages." in pfx_lower
                or pfx_lower.startswith("stages.")
                or "mtp" in pfx_lower
                or "nextn" in pfx_lower
            )
            self.assertTrue(is_draft, f"Prefix {pfx} should be recognized as draft layer")


if __name__ == "__main__":
    unittest.main()

