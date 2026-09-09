import unittest
import sglang_exl3


class TestRegistration(unittest.TestCase):
    def test_quantization_registration(self):
        sglang_exl3.register()
        try:
            from sglang.srt.layers import quantization as sq
            self.assertIn("exl3", sq.QUANTIZATION_METHODS)
        except ImportError:
            self.skipTest("SGLang not installed in local test environment")

    def test_server_args_choices(self):
        sglang_exl3.register()
        try:
            from sglang.srt import server_args as sa
            if hasattr(sa, "QUANTIZATION_CHOICES"):
                self.assertIn("exl3", sa.QUANTIZATION_CHOICES)
        except ImportError:
            self.skipTest("SGLang server_args not importable")


if __name__ == "__main__":
    unittest.main()
