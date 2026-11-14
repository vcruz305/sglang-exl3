import os
import tempfile
import unittest
import torch
try:
    from safetensors.torch import save_file
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

import sglang_exl3
from sglang_exl3.loader import (
    EXL3MMapModelLoader,
    exl3_mmap_weights_iterator,
    is_unified_memory_platform,
    madvise_file_preload,
    LoadFormatEXL3,
    LOAD_FORMAT_EXL3_MMAP,
)


class TestEXL3MMapLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sglang_exl3.register()

    def test_loader_constants_and_platform(self):
        self.assertEqual(LoadFormatEXL3.EXL3_MMAP, "exl3_mmap")
        self.assertEqual(LOAD_FORMAT_EXL3_MMAP, "exl3_mmap")
        is_um = is_unified_memory_platform()
        self.assertIsInstance(is_um, bool)

    @unittest.skipUnless(_HAS_SAFETENSORS, "safetensors not installed")
    def test_mmap_weights_iterator(self):
        tmp_dir = tempfile.mkdtemp()
        shard1_path = os.path.join(tmp_dir, "model-00001-of-00002.safetensors")
        shard2_path = os.path.join(tmp_dir, "model-00002-of-00002.safetensors")

        t1 = torch.randn(64, 128, dtype=torch.float32)
        t2 = torch.randn(128, 64, dtype=torch.float32)
        t3 = torch.randn(32, 32, dtype=torch.float32)

        save_file({"model.layers.0.weight": t1, "model.layers.0.bias": t2}, shard1_path)
        save_file({"model.layers.1.weight": t3}, shard2_path)

        adv_ok = madvise_file_preload(shard1_path)
        self.assertIsInstance(adv_ok, bool)

        loaded = {}
        for name, tensor in exl3_mmap_weights_iterator(
            [shard1_path, shard2_path],
            device="cpu",
            num_threads=2,
            prefetch_shards=2,
        ):
            loaded[name] = tensor

        self.assertEqual(len(loaded), 3)
        self.assertIn("model.layers.0.weight", loaded)
        self.assertIn("model.layers.0.bias", loaded)
        self.assertIn("model.layers.1.weight", loaded)

        torch.testing.assert_close(loaded["model.layers.0.weight"], t1)
        torch.testing.assert_close(loaded["model.layers.0.bias"], t2)
        torch.testing.assert_close(loaded["model.layers.1.weight"], t3)

        try:
            os.remove(shard1_path)
            os.remove(shard2_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

    @unittest.skipUnless(_HAS_SAFETENSORS, "safetensors not installed")
    def test_pread_iterator_roundtrip_dtypes(self):
        from sglang_exl3.loader import iter_safetensors_pread
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "model-00001-of-00001.safetensors")
        tensors = {
            "bf16": torch.randn(33, 17).to(torch.bfloat16),
            "f16": torch.randn(8, 16).to(torch.float16),
            "f32": torch.randn(5).to(torch.float32),
            "i16_trellis": torch.randint(-32768, 32767, (4, 16, 48), dtype=torch.int16),
            "i32": torch.arange(12, dtype=torch.int32).reshape(3, 4),
            "u8": torch.arange(256, dtype=torch.uint8),
            "scalar": torch.tensor(3, dtype=torch.int32),
            "empty": torch.empty(0, 4, dtype=torch.float16),
        }
        save_file(tensors, path)
        try:
            got = dict(iter_safetensors_pread(path))
            self.assertEqual(set(got), set(tensors))
            for name, ref in tensors.items():
                self.assertEqual(got[name].dtype, ref.dtype, name)
                self.assertEqual(tuple(got[name].shape), tuple(ref.shape), name)
                self.assertTrue(torch.equal(got[name], ref), name)
        finally:
            os.remove(path)
            os.rmdir(tmp_dir)

    @unittest.skipUnless(_HAS_SAFETENSORS, "safetensors not installed")
    def test_pread_iterator_fails_closed_on_unknown_dtype(self):
        import struct as _struct
        import json as _json
        from sglang_exl3.loader import iter_safetensors_pread
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "bad.safetensors")
        header = _json.dumps({"x": {"dtype": "C64", "shape": [2], "data_offsets": [0, 16]}}).encode()
        with open(path, "wb") as fh:
            fh.write(_struct.pack("<Q", len(header)) + header + bytes(16))
        try:
            with self.assertRaises(RuntimeError):
                list(iter_safetensors_pread(path))
        finally:
            os.remove(path)
            os.rmdir(tmp_dir)

    def test_sglang_loader_patch(self):
        try:
            from sglang.srt.model_loader.loader import get_model_loader, LoadConfig
            from sglang.srt.configs.model_config import ModelConfig
            
            lc = LoadConfig(load_format="exl3_mmap")
            loader_inst = get_model_loader(lc)
            self.assertIsInstance(loader_inst, EXL3MMapModelLoader)
        except (ImportError, TypeError, AttributeError):
            loader_inst = EXL3MMapModelLoader(load_config=None)
            self.assertEqual(loader_inst.load_format, "exl3_mmap")


if __name__ == "__main__":
    unittest.main()
