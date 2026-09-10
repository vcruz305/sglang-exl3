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
[`glm-5.3-exl3-recipe`](https://github.com/vcruz305/glm-5.3-exl3-recipe)
for GLM-5.3-Flash-EXL3-K2.

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

- **GLM-5.3-Flash-EXL3-K2** (45 layers: 34 KDA linear-attention + 11 DSA
  full-attention, 288 routed experts / 8 active) — primary integration
  target; tested extensively. See the recipe repo above for the tuned
  launch config and measured throughput.
- **Qwen4Exp architecture (`Qwen4ExpForConditionalGeneration` /
  `model_type: qwen4_exp`)** — **not currently loadable.** This is not an
  EXL3-plugin limitation: the checkpoint's reported architecture isn't
  registered in either upstream Transformers or upstream SGLang's model
  registry (`AutoConfig.from_pretrained` fails before the EXL3 quant
  path is ever reached). Needs a model-class implementation upstream (or
  an architecture alias/shim) before this plugin can be tested against
  it.

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
