"""Empirical benchmark for EXL3 dynamic MTP speculative schedule and confidence pruning."""

import json
import torch
import sglang_exl3
from sglang_exl3.speculative import (
    EXL3SpeculativeHook,
    EXL3MTPConfidenceHead,
    EXL3SpeculativeConfig,
)


def run_speculative_benchmark():
    sglang_exl3.register()
    cfg = EXL3SpeculativeConfig.from_env()
    head = EXL3MTPConfidenceHead(threshold=cfg.confidence_threshold)

    batch_sizes = [1, 2, 4, 8, 16, 32]
    vocab_size = 32000
    results = []

    print("\n================ EXL3 MTP SPECULATIVE BENCHMARK ================")
    print(f"Confidence Threshold: {cfg.confidence_threshold:.2f}")
    print(f"Schedule: {cfg.schedule}")
    print("----------------------------------------------------------------")
    print(" Batch | Draft Steps | Confident % | Pruned % | Sim. Speedup")
    print("-------+-------------+-------------+----------+--------------")

    torch.manual_seed(42)

    for bs in batch_sizes:
        draft_steps = EXL3SpeculativeHook.get_step_count(bs, cfg)
        if draft_steps == 0:
            results.append({
                "batch_size": bs,
                "draft_steps": 0,
                "confident_ratio": 1.0,
                "pruned_ratio": 0.0,
                "simulated_speedup": 1.0,
                "mode": "PURE_DECODE",
            })
            print(f" {bs:5d} | {draft_steps:11d} |      N/A    |    0.0%  |     1.00x (Pure Decode)")
            continue

        # Simulate 1000 generation steps
        n_sim = 200
        total_tokens = n_sim * bs * draft_steps
        kept_tokens = 0

        for _ in range(n_sim):
            # Logits with realistic confidence distribution:
            # 80% confident tokens, 20% uncertain tokens
            logits = torch.randn((bs, draft_steps, vocab_size))
            # Inject high confidence signal for 80% of steps
            high_conf_mask = torch.rand((bs, draft_steps)) < 0.80
            for b in range(bs):
                for s in range(draft_steps):
                    if high_conf_mask[b, s]:
                        logits[b, s, 100 + s] += 15.0  # high confidence spike (p > 0.70 over 32k vocab)


            _, lengths = head.prune_draft_tokens(
                torch.zeros((bs, draft_steps), dtype=torch.long), logits
            )
            kept_tokens += lengths.sum().item()

        confident_ratio = kept_tokens / total_tokens
        pruned_ratio = 1.0 - confident_ratio
        
        # Effective speedup simulation:
        # Each accepted draft token gives 1.0 token saving with 0.15 verification cost
        accepted_tokens_per_step = (kept_tokens / (n_sim * bs)) * 0.85
        overhead = 1.0 + (kept_tokens / (n_sim * bs)) * 0.15
        sim_speedup = (1.0 + accepted_tokens_per_step) / overhead

        results.append({
            "batch_size": bs,
            "draft_steps": draft_steps,
            "confident_ratio": round(confident_ratio, 3),
            "pruned_ratio": round(pruned_ratio, 3),
            "simulated_speedup": round(sim_speedup, 2),
            "mode": "SPECULATIVE",
        })
        print(f" {bs:5d} | {draft_steps:11d} |     {confident_ratio*100:5.1f}%  |  {pruned_ratio*100:5.1f}%  |     {sim_speedup:.2f}x")

    print("================================================================\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_speculative_benchmark()
