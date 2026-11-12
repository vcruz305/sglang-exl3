import os
import sys
import unittest
from unittest import mock

import torch

from sglang_exl3 import exl3


class PythonLoopCaptureGuard(unittest.TestCase):
    def _args(self):
        x = torch.zeros(1, 4)
        ids = torch.full((1, 2), -1, dtype=torch.long)  # every slot invalid: loop body never runs
        w = torch.zeros(1, 2)
        return x, ids, w

    def test_refuses_while_stream_is_capturing(self):
        x, ids, w = self._args()
        with mock.patch.object(exl3.torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "not CUDA-graph capturable"):
                exl3.apply_exl3_python_loop(x, ids, w, [], None, float("inf"))

    def test_eager_path_still_runs(self):
        x, ids, w = self._args()
        with mock.patch.object(exl3.torch.cuda, "is_current_stream_capturing", return_value=False):
            out = exl3.apply_exl3_python_loop(x, ids, w, [], None, float("inf"))
        self.assertEqual(tuple(out.shape), (1, 4))

    def test_cpu_only_torch_is_not_treated_as_capture(self):
        x, ids, w = self._args()
        with mock.patch.object(exl3.torch.cuda, "is_current_stream_capturing", side_effect=RuntimeError("no cuda")):
            self.assertFalse(exl3._cuda_graph_capture_active())
            out = exl3.apply_exl3_python_loop(x, ids, w, [], None, float("inf"))
        self.assertEqual(tuple(out.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
