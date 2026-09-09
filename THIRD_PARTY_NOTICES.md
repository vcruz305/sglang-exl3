# Third-party notices

This project is licensed under Apache-2.0. It contains and derives from MIT-licensed work by other authors, whose
copyright and permission notices are reproduced below as those licenses require.

---

## Mia's AI Lab, GLM-5.3-Flash-EXL3-2x-DGX-Sparks

Authors: Mia's AI Lab ([@MiaAI-Lab](https://github.com/MiaAI-Lab)) and
[@plotarmordev](https://github.com/plotarmordev).

https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks

**Files copied from that project into this one:**

- `csrc/exl3_fat_gemm.cu` and `csrc/exl3_fat_gemm.cuh` are taken from `overlay/exl3_fat_gemm.cu`
  and `overlay/exl3_fat_gemm.cuh` at commit `4b8d3c7`, "perf(exl3): accelerate fat-expert prefill".
  The only modification is `#include` paths adjusted for this project's layout. The kernel is
  otherwise unchanged.

**Files in this project substantially derived from that project:**

- `src/sglang_exl3/exl3.py` derives from `overlay/exl3.py`, first published there on 2026-08-27,
  which precedes this project's first commit. Substantial portions of the routed-expert EXL3/MCG
  path, including its pointer-table construction, expert-map pinning, and diagnostic strings,
  originate there.

```
MIT License

Copyright (c) 2026 Mia's AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Turboderp, ExLlamaV3

Author: [@turboderp](https://github.com/turboderp).

https://github.com/turboderp/exllamav3

The CUDA sources in `csrc/` build against ExLlamaV3's extension headers, including
`quant/exl3_gemv_kernel.cuh`, `exl3_dq.cuh`, `hadamard_inner.cuh`, `util.h`, `util.cuh` and
`ptx.cuh`, and the GEMV and batched kernels reuse its kernel body. The EXL3 trellis format, the MCG
codebook, and the quantization method itself are ExLlamaV3's work.

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
