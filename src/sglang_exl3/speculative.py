"""Native SGLang speculative runner hook with MTP confidence head and dynamic schedule.

Implements:
1. Dynamic batch-aware speculative schedule for sm_121 Blackwell:
   - BS [1..4]  -> 3 draft tokens (latency-dominated)
   - BS [5..8]  -> 2 draft tokens (balanced)
   - BS [9..16] -> 1 draft token  (throughput-dominated)
   - BS > 16    -> 0 draft tokens (pure batched decode)
2. Confidence-guided candidate pruning:
   - Early-truncates multi-token speculative branches when cumulative top-1
     probability drops below confidence threshold, avoiding wasted verification GEMMs.
3. SGLang speculative runner hook integration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .exl3 import (
    DEFAULT_SPECULATIVE_SCHEDULE,
    SPECULATIVE_SCHEDULE_ENV,
    ADAPTIVE_VERIFICATION_ENV,
    get_speculative_draft_tokens,
    parse_speculative_schedule,
    filter_speculative_candidates,
    is_adaptive_verification_enabled,
)

logger = logging.getLogger("sglang.exl3.speculative")

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
CONFIDENCE_THRESHOLD_ENV = "SGLANG_EXL3_CONFIDENCE_THRESHOLD"


@dataclass
class EXL3SpeculativeConfig:
    schedule: list[tuple[int, int, int]]
    adaptive_verification: bool = True
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD

    @classmethod
    def from_env(cls) -> "EXL3SpeculativeConfig":
        raw_schedule = os.environ.get(SPECULATIVE_SCHEDULE_ENV)
        schedule = (
            parse_speculative_schedule(raw_schedule)
            if raw_schedule
            else list(DEFAULT_SPECULATIVE_SCHEDULE)
        )
        adaptive = is_adaptive_verification_enabled()
        raw_thresh = os.environ.get(CONFIDENCE_THRESHOLD_ENV)
        try:
            thresh = float(raw_thresh) if raw_thresh else DEFAULT_CONFIDENCE_THRESHOLD
        except ValueError:
            thresh = DEFAULT_CONFIDENCE_THRESHOLD
        return cls(schedule=schedule, adaptive_verification=adaptive, confidence_threshold=thresh)


class EXL3MTPConfidenceHead:
    """Evaluates multi-token prediction confidence and performs prefix-tree candidate pruning."""

    def __init__(self, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def evaluate_logits(
        self, logits: torch.Tensor, threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given draft logits of shape (batch, draft_steps, vocab_size), compute:
        
        Returns:
            top_tokens: (batch, draft_steps) token IDs
            probs: (batch, draft_steps) top-1 probabilities
            valid_mask: (batch, draft_steps) boolean mask of confident prefix tokens
        """
        thresh = self.threshold if threshold is None else threshold
        probs_all = F.softmax(logits.float(), dim=-1)
        probs, top_tokens = torch.max(probs_all, dim=-1)
        valid_mask, _ = filter_speculative_candidates(probs, threshold=thresh)
        return top_tokens, probs, valid_mask

    def prune_draft_tokens(
        self, draft_tokens: torch.Tensor, logits: torch.Tensor, threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prunes low-confidence tail tokens from draft token sequence."""
        _, _, valid_mask = self.evaluate_logits(logits, threshold=threshold)
        # Keep only the valid prefix; count per sequence
        lengths = valid_mask.sum(dim=-1, dtype=torch.long)
        return valid_mask, lengths


class EXL3SpeculativeHook:
    """Manages integration hooks into SGLang's speculative runtime workers."""

    _HOOK_APPLIED = False

    @classmethod
    def get_step_count(cls, batch_size: int, config: Optional[EXL3SpeculativeConfig] = None) -> int:
        sched = config.schedule if config else None
        return get_speculative_draft_tokens(batch_size, schedule=sched)

    @classmethod
    def apply_hooks(cls) -> bool:
        """Hook into SGLang speculative runner workers if available."""
        if cls._HOOK_APPLIED:
            return True

        patched_any = False
        # 1. Patch FrozenKVMTPDraftWorker / FrozenKVMTPWorkerV2 if imported
        try:
            from sglang.srt.speculative import frozen_kv_mtp_worker_v2 as mtp_mod

            if hasattr(mtp_mod, "FrozenKVMTPDraftWorker"):
                orig_forward = mtp_mod.FrozenKVMTPDraftWorker.forward
                confidence_head = EXL3MTPConfidenceHead()

                def exl3_mtp_forward(self, *args, **kwargs):
                    output = orig_forward(self, *args, **kwargs)
                    if getattr(self, "_exl3_adaptive_prune", True) and hasattr(output, "logits"):
                        try:
                            mask, lengths = confidence_head.prune_draft_tokens(
                                output.draft_tokens, output.logits
                            )
                            output.exl3_valid_mask = mask
                            output.exl3_draft_lengths = lengths
                        except Exception as exc:
                            logger.debug("EXL3 adaptive confidence pruning skipped: %s", exc)
                    return output

                mtp_mod.FrozenKVMTPDraftWorker.forward = exl3_mtp_forward
                patched_any = True
                logger.info("Hooked EXL3MTPConfidenceHead into FrozenKVMTPDraftWorker")
        except (ImportError, AttributeError):
            pass

        # 2. Patch AdaptiveController if present
        try:
            from sglang.srt.speculative import adaptive_runtime_state as adapt_mod

            if hasattr(adapt_mod, "AdaptiveController"):
                orig_update = adapt_mod.AdaptiveController.update

                def exl3_adaptive_update(self, batch_size, *args, **kwargs):
                    # Use sm_121 Blackwell native schedule
                    spec_steps = EXL3SpeculativeHook.get_step_count(batch_size)
                    res = orig_update(self, batch_size, *args, **kwargs)
                    if hasattr(self, "current_spec_steps"):
                        self.current_spec_steps = min(self.current_spec_steps, spec_steps)
                    return res

                adapt_mod.AdaptiveController.update = exl3_adaptive_update
                patched_any = True
                logger.info("Hooked EXL3 batch schedule into SGLang AdaptiveController")
        except (ImportError, AttributeError):
            pass

        cls._HOOK_APPLIED = True
        return patched_any
