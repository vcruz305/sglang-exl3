# sglang-exl3

A native EXL3 (ExLlamaV3 trellis/MCG-codebook) quantization plugin for
[SGLang](https://github.com/sgl-project/sglang), targeting NVIDIA Blackwell
`sm_121` (GB10/DGX Spark class hardware) with a compiled batched MoE kernel
and Dynamic FP8 DSpark-stage support.

This repo contains the **plugin itself** — the SGLang model-loader hook,
quantization method registration, and CUDA kernels. It is model-agnostic:
it has been run against dense, MoE, and hybrid linear/full-attention
architectures. For a specific model's tuned serving recipe (launch flags,
speculative-decoding config, measured throughput, and the full
investigation trail behind those numbers), see the separate recipe repo
for that model — e.g.
[`GLM-5.3-Flash-EXL3-K2-SGLang-DGX-Spark-recipe`](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-SGLang-DGX-Spark-recipe)
for GLM-5.3-Flash-EXL3-K2, or the model-coverage table below for all
tested models. Ripple-65B-Preview with DFlash2 speculative decoding has
its own recipe repo:
[`Ripple-65B-Preview-EXL3-SGLang-recipe`](https://github.com/vcruz305/Ripple-65B-Preview-EXL3-SGLang-recipe).

## What this plugin provides

- `src/sglang_exl3/exl3.py` — `EXL3Config`, `EXL3LinearMethod`,
  `EXL3MoEMethod`: SGLang quantization-method registration for EXL3
  checkpoints (routed-expert MoE and dense linear layers).
- `src/sglang_exl3/loader.py` — `EXL3MMapModelLoader`: mmap-based weight
  loading for EXL3 checkpoints (`--load-format exl3_mmap`).
- `src/sglang_exl3/speculative.py` — EAGLE-style speculative-decoding
  hooks compatible with EXL3-quantized draft/target model pairs.
- `src/sglang_exl3/ops.py`, `src/sglang_exl3/hadamard_cache.py` — custom
  op registration and Hadamard-transform caching used by the EXL3 GEMM/MoE
  kernels.
- `csrc/` — CUDA kernels: `exl3_gemm`/`exl3_gemv` (single-matrix EXL3
  matmul, dispatched by batch size), `exl3_fat_gemm` (fused gate+up
  projection), `p2b_moe`/`p2b_batched` (native batched MoE dispatch: one
  kernel launch handles the whole active-expert set for a token batch,
  not a per-expert loop). Kernel-shape selection is architecture-aware
  (`CC_AMPERE`/`CC_ADA`/`CC_HOPPER`/`CC_BLACKWELL` branches in
  `exl3_kernel_map.cu`), not hardcoded to one GPU generation.
- `scripts/server_supervisor.py` — a generic, ownership-safe process
  supervisor for any SGLang server launch on a memory-constrained box:
  computes an admission reserve from `--model-gib`/`--draft-gib`/
  `--kv-gib`/`--graph-gib`, enforces a live memory-kill trip threshold,
  and reconciles a dropped SSH session via a durable status file (does
  not discover/kill processes by name — only by owned PID + process
  group + start-time). Not GLM- or Qwen-specific; use it to wrap any
  model's launch command.
- `scripts/reconcile_supervised_run.sh` — read-only status check against
  a running supervised server (see above).
- `scripts/generate_manifest.py`, `scripts/audit_tensor_inventory.py`,
  `scripts/diagnose_moe.py` — checkpoint inspection utilities (tensor
  naming/shape audits, MoE-layer diagnostics) usable against any EXL3
  checkpoint, not tied to one model family.
- `scripts/install_exllamav3_aarch64.sh`, `scripts/patch_exllamav3_aarch64.py`
  — build/patch helpers for getting the upstream `exllamav3` package
  working on aarch64 (e.g. DGX Spark / GB10).
- `benchmarks/` — generic harnesses (concurrency, CUDA graph capture,
  Hadamard cache, mmap load, speculative decoding, live serving) that
  take a model path as an argument rather than hardcoding one.
- `tests/` — unit tests for the above (config parsing, CUDA graph
  capture, mmap loader, MoE dispatch, quantization-method registration,
  speculative decoding, the supervisor).

## Known model-architecture coverage

Numbers below are **measured**, single DGX Spark GB10 (sm_121), from the
model's own recipe repo — see each recipe for full launch flags,
hardware notes, and the investigation trail behind the numbers.

| Model | Quant | Layout | MTP / speculative | Decode tok/s | Recipe |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash-EXL3-K2 | EXL3, ~91 GiB | 45 layers: 34 KDA linear-attn + 11 DSA full-attn, 288 routed experts / 8 active | k=2–16 MTP | ~16 tok/s @ 8k ctx | [vLLM recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) · [SGLang recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-SGLang-DGX-Spark-recipe) |
| Qwen3.8-Flash-Next-EXL3 | EXL3, 3.05 bpw, ~87 GiB | `Qwen4ExpForConditionalGeneration` (`model_type: qwen4_exp`), QSA sparse attention | EAGLE/MTP, native-EXL3 draft (`steps=3 topk=1 draft_tokens=4`) | **~22–31 tok/s with MTP** (peak 35), **~17 tok/s baseline (no MTP)** | [SGLang recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-SGLang-DGX-Spark-recipe) · [vLLM recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) |
| [Ripple-65B-Preview-SAGE-REXL3](https://huggingface.co/vcruz305/Ripple-65B-Preview-SAGE-REXL3-5.41bpw) | SAGE-REXL3, [`5.41 bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-SAGE-REXL3-5.41bpw) (~43 GB) · [`2.57 bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-SAGE-REXL3-2.57bpw) · [`1.95 bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-SAGE-REXL3-1.95bpw) | `RicursiveForCausalLM`: 80 layers, hybrid full-attn (every 5th) + sliding-window (1024), NoPE, value residuals, PeriNorm | [DFlash2 draft](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3) — EXL3 [`4.55bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3/tree/main/4.55bpw) (recommended) / [`3.60bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3/tree/main/3.60bpw) / [`3.08bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3/tree/main/3.08bpw), or [BF16](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) (`num_draft_tokens=16`) | **~64.8 tok/s greedy** / **~53.1 tok/s t=0.6** decode, **~1,800 tok/s prefill**, accept len 2.05–3.55, 117,376-token full KV pool on 1× RTX PRO 6000 96 GB | [SGLang recipe](https://github.com/vcruz305/Ripple-65B-Preview-EXL3-SGLang-recipe) |

**Qwen4Exp / Qwen3.8-Flash-Next details:**

- Architecture is registered upstream in both Transformers and SGLang as
  of this plugin's `mul1`-codebook fix ([PR #1](https://github.com/vcruz305/sglang-exl3/pull/1),
  merged `20bce5e`) — the earlier "not currently loadable" limitation
  (missing MoE codebook multiplier handling for routed EXL3 experts) is
  resolved.
- EAGLE/MTP speculative decoding works via a second fix
  ([PR #2](https://github.com/vcruz305/sglang-exl3/pull/2), merged) that
  loads the MTP draft's MoE/linear layers as native EXL3 instead of
  falling back to random-initialized dense/BF16 weights. Before this
  fix: accept rate 0.00, accept len 1.00 (draft never contributed).
  After: **accept rate 0.47–0.72, accept len 2.4–3.2**.
- `speculative_eagle_topk` is **hard-capped at 1** for this architecture
  (`NotImplementedError` above that) — not a tunable flag, confirmed via
  direct testing of `topk=4`.
- `speculative_num_steps` above 3 (e.g. `steps=4`–`6`) hits a CUDA
  device-side assert in the QSA indexer's shared sparse-index buffer
  (`QSAMTPSharedSparseIndices.tail_width`, sized for `steps=3` only) —
  a runtime buffer-sizing bug, not a config issue. `steps=3` is the
  current ceiling.
- **Known open issue**: server becomes unreachable (`CUDA error:
  device-side assert triggered` in the QSA indexer's gather kernel)
  after ~60–70 minutes idle; the first request after the idle gap
  crashes it. Reproduced deterministically. Root cause not yet
  identified — see the recipe repo for current status before relying on
  long-idle-then-serve workloads.

**Ripple / Ripple-65B-Preview + DFlash2 details:**

- Adds `RicursiveForCausalLM` — quantization-aware Ripple target (separate `q/k/v/g/gate/up` projections, PeriNorm decoder, sliding-window + NoPE with full attention every 5th layer, value residuals) — and `DFlash2DraftModel` EXL3/BF16 draft support (`src/sglang_exl3/models/ricursive.py`, `src/sglang_exl3/models/dflash_exl3.py`). Measured on SGLang `0.5.19`, one RTX PRO 6000 Blackwell 96 GB (`sm_120`), with the [`5.41bpw` SAGE target](https://huggingface.co/vcruz305/Ripple-65B-Preview-SAGE-REXL3-5.41bpw) (companion packs: [`2.57bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-SAGE-REXL3-2.57bpw), [`1.95bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-SAGE-REXL3-1.95bpw)) + [`4.55bpw` EXL3 DFlash2 draft](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3/tree/main/4.55bpw): **~64.8 tok/s greedy decode** (44.9 @ 1,564-token prompt, 49.3 @ 3,152), **~53.1 tok/s at `t=0.6`** (41.2 / 51.3), cold prefill **~1,096–1,802 tok/s**, DFlash accept len **2.05–3.55**, and 117,376 full-attention / 93,888 SWA KV tokens at `--mem-fraction-static 0.82` with the 16-token DFlash block.
- The DFlash2 draft packs ([`vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3`](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3) at [`4.55bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3/tree/main/4.55bpw) / [`3.60bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3/tree/main/3.60bpw) / [`3.08bpw`](https://huggingface.co/vcruz305/Ripple-65B-Preview-DFlash2-SAGE-REXL3/tree/main/3.08bpw), plus [BF16 (`ricursiveAI/Ripple-65B-Preview-DFlash2`)](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2)) load as native EXL3 trellis groups; the BF16 draft's stacked `q/k/v → qkv_proj` and `gate/up → gate_up_proj` weights are mapped in `dflash_exl3.py`'s `load_weights`. Skipping that map collapses acceptance to 1.00 (~19.4 tok/s).
- Draft `fc` and `down_proj` use selective FP32 dynamic scaling (`_exl3_scale_input`); omitting it overflows FP16 on long prompts (accept len 1.00). The 80-layer target's 640 linears pay zero scaling cost.
- `SGLANG_EXL3_MGEMM=1` (default) fuses `(q_proj, g_proj)` (80/80 layers), `(k_proj, v_proj)` with value residuals (80/80 layers), `(gate_proj, up_proj)` (79/80 layers), and the draft's `gate_up_proj` + k/v shards into multi-matrix `exl3_mgemm` launches (cos ≈ 0.9999998 vs per-linear reference). The 1D-flat contiguous CUDA-graph workspace never reallocates during capture.
- Full launch flags, tuning knobs, SixCat eval tables, and troubleshooting live in the [Ripple-65B-Preview-EXL3-SGLang-recipe](https://github.com/vcruz305/Ripple-65B-Preview-EXL3-SGLang-recipe) repo (pin `5d1c9ef`).

## Install

See `scripts/install_exllamav3_aarch64.sh` for the aarch64/GB10 build
path (upstream `exllamav3` + this plugin's compiled extension). Then:

```bash
pip install -e .
```

The plugin self-registers into SGLang on import — set
`SGLANG_EXL3_REGISTER=1` and either import `sglang_exl3` before launching,
or use `python -m sglang_exl3.launch_server` in place of
`python -m sglang.launch_server`.

## License

Apache-2.0. See `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` — some
kernel code originates from [Mia's AI Lab's
GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
project, attributed in the relevant `csrc/` file headers.

## Contributing

Adding a script? Read `CONTRIBUTING.md` first — check for existing
tooling in `scripts/`/`tests`/`benchmarks/` before writing a new one.
