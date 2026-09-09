import os
import unittest
import torch
import sglang_exl3
from sglang_exl3.hadamard_cache import (
    HadamardPrefixCache,
    transform_hadamard_fast,
    is_moe_quant_once_enabled,
    OPT_MOE_QUANT_ONCE_ENV,
)


class TestHadamardCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sglang_exl3.register()

    def test_hadamard_transform_and_involution(self):
        # Walsh-Hadamard transform is symmetric and self-inverse (involution)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn((2, 256), dtype=torch.float32, device=device)

        # First transform
        h1 = transform_hadamard_fast(x)
        self.assertEqual(h1.shape, x.shape)
        self.assertEqual(h1.dtype, x.dtype)

        # Second transform should invert back to original x
        h2 = transform_hadamard_fast(h1)
        torch.testing.assert_close(h2, x, rtol=1e-3, atol=1e-3)

    def test_cache_store_lookup_and_slicing(self):
        cache = HadamardPrefixCache(max_entries=4)
        prefix_hash = 0xABCD1234
        layer_id = 5

        data = torch.randn((128, 4096), dtype=torch.float16)
        cache.put(prefix_hash, layer_id, data)

        # Exact length match
        hit_exact = cache.get(prefix_hash, layer_id, 128)
        self.assertIsNotNone(hit_exact)
        self.assertEqual(hit_exact.shape, torch.Size([128, 4096]))
        torch.testing.assert_close(hit_exact, data)

        # Prefix slice match (shorter sequence length query)
        hit_slice = cache.get(prefix_hash, layer_id, 64)
        self.assertIsNotNone(hit_slice)
        self.assertEqual(hit_slice.shape, torch.Size([64, 4096]))
        torch.testing.assert_close(hit_slice, data[:64])

        # Miss on longer length
        miss_long = cache.get(prefix_hash, layer_id, 256)
        self.assertIsNone(miss_long)

        # Miss on different layer
        miss_layer = cache.get(prefix_hash, 6, 128)
        self.assertIsNone(miss_layer)

        # Miss on different hash
        miss_hash = cache.get(0x99999999, layer_id, 128)
        self.assertIsNone(miss_hash)

    def test_lru_eviction(self):
        cache = HadamardPrefixCache(max_entries=2)
        cache.put(1, 0, torch.zeros((10, 128)))
        cache.put(2, 0, torch.zeros((10, 128)))
        
        # Access 1 to make it most recently used
        _ = cache.get(1, 0, 10)
        
        # Put 3 -> should evict 2
        cache.put(3, 0, torch.zeros((10, 128)))
        
        self.assertIsNotNone(cache.get(1, 0, 10))
        self.assertIsNone(cache.get(2, 0, 10))
        self.assertIsNotNone(cache.get(3, 0, 10))

    def test_env_toggle(self):
        orig = os.environ.get(OPT_MOE_QUANT_ONCE_ENV)
        try:
            os.environ[OPT_MOE_QUANT_ONCE_ENV] = "0"
            self.assertFalse(is_moe_quant_once_enabled())
            cache = HadamardPrefixCache()
            cache.put(1, 0, torch.zeros((10, 128)))
            self.assertIsNone(cache.get(1, 0, 10))

            os.environ[OPT_MOE_QUANT_ONCE_ENV] = "1"
            self.assertTrue(is_moe_quant_once_enabled())
        finally:
            if orig is not None:
                os.environ[OPT_MOE_QUANT_ONCE_ENV] = orig
            else:
                os.environ.pop(OPT_MOE_QUANT_ONCE_ENV, None)


if __name__ == "__main__":
    unittest.main()
