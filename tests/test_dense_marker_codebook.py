import unittest

import torch

from sglang_exl3.dense import EXL3DenseLinear
from sglang_exl3.exl3 import MCG_MARKER_SIGNED_INT32, MUL1_MARKER_SIGNED_INT32


def _linear(mcg=None, mul1=None):
    # K=3 trellis: last dim 16*K = 48.
    trellis = torch.zeros(2, 2, 48, dtype=torch.int16)
    suh = torch.ones(32, dtype=torch.float16)
    svh = torch.ones(32, dtype=torch.float16)
    m = None if mcg is None else torch.tensor([mcg], dtype=torch.int32)
    u = None if mul1 is None else torch.tensor([mul1], dtype=torch.int32)
    return EXL3DenseLinear(trellis, suh, svh, m, u)


class TestDenseCodebookMarkers(unittest.TestCase):
    def test_mul1_marker_selects_mul1_only(self):
        lin = _linear(mul1=MUL1_MARKER_SIGNED_INT32)
        self.assertTrue(lin.mul1)
        self.assertFalse(lin.mcg)

    def test_mcg_marker_selects_mcg_only(self):
        lin = _linear(mcg=MCG_MARKER_SIGNED_INT32)
        self.assertTrue(lin.mcg)
        self.assertFalse(lin.mul1)

    def test_no_marker_is_plain_3inst_cb0_as_in_exllamav3(self):
        # exllamav3 leaves cb=0 when neither flag is set (exl3_gemm.cu:174-176);
        # the plugin must not silently force mcg here.
        lin = _linear()
        self.assertFalse(lin.mcg)
        self.assertFalse(lin.mul1)


if __name__ == "__main__":
    unittest.main()
