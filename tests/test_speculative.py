import unittest
import torch
import sglang_exl3
from sglang_exl3 import speculative

EXL3SpeculativeConfig = speculative.EXL3SpeculativeConfig
EXL3MTPConfidenceHead = speculative.EXL3MTPConfidenceHead
EXL3SpeculativeHook = speculative.EXL3SpeculativeHook



class TestEXL3Speculative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sglang_exl3.register()

    def test_dynamic_schedule_steps(self):
        # Default schedule: (1, 4, 3), (5, 8, 2), (9, 16, 1)
        self.assertEqual(EXL3SpeculativeHook.get_step_count(1), 3)
        self.assertEqual(EXL3SpeculativeHook.get_step_count(4), 3)
        self.assertEqual(EXL3SpeculativeHook.get_step_count(5), 2)
        self.assertEqual(EXL3SpeculativeHook.get_step_count(8), 2)
        self.assertEqual(EXL3SpeculativeHook.get_step_count(9), 1)
        self.assertEqual(EXL3SpeculativeHook.get_step_count(16), 1)
        self.assertEqual(EXL3SpeculativeHook.get_step_count(17), 0)
        self.assertEqual(EXL3SpeculativeHook.get_step_count(32), 0)

    def test_confidence_head_pruning(self):
        head = EXL3MTPConfidenceHead(threshold=0.70)
        vocab_size = 100

        # Case 1: High confidence throughout (all 3 steps confident)
        logits1 = torch.zeros((1, 3, vocab_size))
        logits1[0, 0, 42] = 10.0  # prob ~ 1.0
        logits1[0, 1, 15] = 10.0  # prob ~ 1.0
        logits1[0, 2, 77] = 10.0  # prob ~ 1.0

        top_tokens, probs, valid_mask = head.evaluate_logits(logits1)
        self.assertTrue(torch.all(valid_mask))
        self.assertEqual(valid_mask.sum().item(), 3)
        self.assertEqual(top_tokens.tolist(), [[42, 15, 77]])

        # Case 2: Low confidence on step 2 (index 1)
        # Should prune step 2 AND step 3
        logits2 = torch.zeros((1, 3, vocab_size))
        logits2[0, 0, 42] = 10.0  # confident
        logits2[0, 1, :] = 0.0   # uniform across 100 classes (prob 0.01 < 0.70)
        logits2[0, 2, 77] = 10.0  # confident, but tail after break

        _, _, valid_mask2 = head.evaluate_logits(logits2)
        self.assertEqual(valid_mask2[0].tolist(), [True, False, False])

        # Case 3: Low confidence on step 1 (index 0)
        # Should prune everything
        logits3 = torch.zeros((1, 3, vocab_size))
        logits3[0, 0, :] = 0.0   # low confidence
        logits3[0, 1, 15] = 10.0
        logits3[0, 2, 77] = 10.0

        _, _, valid_mask3 = head.evaluate_logits(logits3)
        self.assertEqual(valid_mask3[0].tolist(), [False, False, False])

    def test_multi_batch_pruning(self):
        head = EXL3MTPConfidenceHead(threshold=0.60)
        vocab_size = 50
        batch_size = 2
        steps = 3

        logits = torch.zeros((batch_size, steps, vocab_size))
        # Batch 0: confident all 3 steps
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        # Batch 1: drops at step 2
        logits[1, 0, 1] = 10.0
        logits[1, 1, :] = 0.0
        logits[1, 2, 3] = 10.0

        draft_tokens = torch.tensor([[1, 2, 3], [1, 5, 3]])
        mask, lengths = head.prune_draft_tokens(draft_tokens, logits)

        self.assertEqual(lengths.tolist(), [3, 1])
        self.assertEqual(mask[0].tolist(), [True, True, True])
        self.assertEqual(mask[1].tolist(), [True, False, False])

    def test_speculative_hook_registration(self):
        applied = EXL3SpeculativeHook.apply_hooks()
        self.assertIsInstance(applied, bool)
        cfg = EXL3SpeculativeConfig.from_env()
        self.assertEqual(len(cfg.schedule), 3)
        self.assertEqual(cfg.confidence_threshold, 0.70)


if __name__ == "__main__":
    unittest.main()
