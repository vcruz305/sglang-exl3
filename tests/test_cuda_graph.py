import unittest
import torch
import sglang_exl3
from sglang_exl3 import ops


class TestCUDAGraphAndCustomOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sglang_exl3.register()
        ops.register_custom_ops()

    def test_ops_registered_in_torch(self):
        self.assertTrue(hasattr(torch.ops, "sglang_exl3"))
        expected_ops = [
            "dequant_trellis",
            "exl3_gemv",
            "exl3_gemm",
            "p2b_gemv_batched",
            "p2b_fused_moe",
            "exl3_fat_gemm",
            "exl3_fat_gemm_scatter",
        ]
        for op_name in expected_ops:
            self.assertTrue(
                hasattr(torch.ops.sglang_exl3, op_name),
                f"Missing registered custom op: torch.ops.sglang_exl3.{op_name}",
            )

    def test_meta_fake_tensor_dispatch(self):
        try:
            from torch._subclasses.fake_tensor import FakeTensorMode
        except ImportError:
            self.skipTest("FakeTensorMode not available in this PyTorch version")

        with FakeTensorMode():
            x = torch.empty((2, 128), dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
            trellis = torch.empty((8, 16, 2), dtype=torch.int16, device=x.device)
            suh = torch.empty((128,), dtype=torch.float16, device=x.device)
            svh = torch.empty((256,), dtype=torch.float16, device=x.device)

            out_gemv = torch.ops.sglang_exl3.exl3_gemv(x, trellis, suh, svh, 2, True, 1)
            self.assertEqual(out_gemv.shape, torch.Size([2, 256]))
            self.assertEqual(out_gemv.dtype, torch.float16)

            out_gemm = torch.ops.sglang_exl3.exl3_gemm(x, trellis, suh, svh, 2, True)
            self.assertEqual(out_gemm.shape, torch.Size([2, 256]))
            self.assertEqual(out_gemm.dtype, torch.float16)

            out_dequant = torch.ops.sglang_exl3.dequant_trellis(trellis, suh, svh, 2, True)
            self.assertEqual(out_dequant.shape, torch.Size([128, 256]))
            self.assertEqual(out_dequant.dtype, torch.float16)

            # Test p2b_fused_moe fake tensor dispatch
            fused_out = torch.empty((1, 128), dtype=torch.float16, device=x.device)
            fake_ptrs = torch.empty((4,), dtype=torch.int64, device=x.device)
            fake_ids = torch.empty((6,), dtype=torch.int32, device=x.device)
            fake_weights = torch.empty((6,), dtype=torch.float16, device=x.device)
            res_moe = torch.ops.sglang_exl3.p2b_fused_moe(
                x[:1, :128],
                fused_out,
                fake_ptrs, fake_ptrs, fake_ptrs,
                fake_ptrs, fake_ptrs, fake_ptrs,
                fake_ptrs, fake_ptrs, fake_ptrs,
                fake_ids, fake_weights,
                2, 2, 2, True,
            )
            self.assertIsNone(res_moe)

    def test_cuda_graph_capture_replay(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available for CUDA Graph test")

        device = torch.device("cuda")
        x = torch.randn((1, 128), dtype=torch.float16, device=device)
        trellis = torch.zeros((8, 16, 2), dtype=torch.int16, device=device)
        suh = torch.ones((128,), dtype=torch.float16, device=device)
        svh = torch.ones((256,), dtype=torch.float16, device=device)

        # Warmup stream
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                _ = torch.ops.sglang_exl3.exl3_gemv(x, trellis, suh, svh, 2, True, 1)
        torch.cuda.current_stream(device=device).wait_stream(stream)

        # Capture CUDA Graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out_static = torch.ops.sglang_exl3.exl3_gemv(x, trellis, suh, svh, 2, True, 1)

        # Replay CUDA Graph
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(out_static.shape, torch.Size([1, 256]))
        self.assertEqual(out_static.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
