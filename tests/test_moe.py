import unittest
import torch
import torch.nn as nn
from sglang_exl3.exl3 import (
    EXL3Config,
    EXL3MoEMethod,
    MCG_MARKER_SIGNED_INT32,
    map_topk_to_local,
    compute_mla_kv_cache_bytes,
    validate_context_scaling,
)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()


class TestEXL3MoE(unittest.TestCase):
    def test_weight_creation(self):
        cfg = EXL3Config(bits=3)
        method = EXL3MoEMethod(cfg, bits=3)
        layer = DummyLayer()

        num_experts = 4
        hidden_size = 4096
        intermediate_size = 2048

        method.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.float16,
        )

        self.assertTrue(hasattr(layer, "w13_trellis"))
        self.assertTrue(hasattr(layer, "w13_suh"))
        self.assertTrue(hasattr(layer, "w13_svh"))
        self.assertTrue(hasattr(layer, "w13_mcg"))
        self.assertTrue(hasattr(layer, "w2_trellis"))
        self.assertTrue(hasattr(layer, "w2_suh"))
        self.assertTrue(hasattr(layer, "w2_svh"))
        self.assertTrue(hasattr(layer, "w2_mcg"))

        # Shape validation: 3 bits -> k_words = 48
        # in_tiles = 4096 // 16 = 256, out_tiles = 2048 // 16 = 128
        self.assertEqual(tuple(layer.w13_trellis.shape), (4, 2, 256, 128, 48))
        self.assertEqual(tuple(layer.w2_trellis.shape), (4, 128, 256, 48))

    def test_map_topk_to_local(self):
        ids = torch.tensor([[0, 1], [2, 5]], dtype=torch.long)
        mapped = map_topk_to_local(ids, n_local=4, expert_map=None)
        # ID 5 is non-local -> mapped to sentinel 4
        self.assertEqual(mapped[1, 1].item(), 4)
        self.assertEqual(mapped[0, 0].item(), 0)

    def test_context_scaling_mla(self):
        # FP8 MLA: 43 layers, 512 latent + 64 rope = 576 bytes/token
        bytes_64k = compute_mla_kv_cache_bytes(65536, num_layers=43, kv_lora_rank=512, qk_rope_head_dim=64)
        self.assertEqual(bytes_64k, 65536 * 43 * 576)

        # 256k context check on 128 GB Unified Memory
        res_256k = validate_context_scaling(262144, model_weights_gb=95.4, total_mem_gb=128.0, mem_util=0.90)
        self.assertTrue("fits" in res_256k)
        self.assertGreater(res_256k["kv_cache_gb"], 5.0)


if __name__ == "__main__":
    unittest.main()
