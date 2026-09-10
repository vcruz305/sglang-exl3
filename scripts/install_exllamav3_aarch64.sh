#!/usr/bin/env bash
set -euo pipefail

# Locate Python / Virtual Environment
VENV="${VENV:-$HOME/venvs/sglang-exl3}"
PYTHON="${VENV}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Error: Python not found at $PYTHON"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configure CUDA PATH and nvcc
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda-13.0/bin /usr/local/cuda/bin; do
    if [[ -x "$d/nvcc" ]]; then export PATH="$d:$PATH"; echo "Added $d to PATH"; break; fi
  done
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export MAX_JOBS=8

echo "=== Building ExLlamaV3 with ARM64 AArch64 Intrinsics Patch ==="
echo "Target venv: $VENV"
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

# 1. Locate, clone, or reuse ExLlamaV3 source
EXLLAMAV3_SRC="${EXLLAMAV3_SRC:-/tmp/exllamav3_build}"
EXLLAMAV3_REV="${EXLLAMAV3_REV:-master}"

if [[ ! -f "$EXLLAMAV3_SRC/setup.py" ]]; then
  if [[ -f "$HOME/src/exllamav3/setup.py" ]]; then
    EXLLAMAV3_SRC="$HOME/src/exllamav3"
    echo "Found existing source at $EXLLAMAV3_SRC"
  else
    rm -rf "$EXLLAMAV3_SRC"
    echo "Cloning canonical ExLlamaV3 from turboderp-org..."
    git clone --depth 1 "https://github.com/turboderp-org/exllamav3.git" "$EXLLAMAV3_SRC"
  fi
fi

# 2. Apply ARM64 intrinsics patch
echo "Applying ARM64 patch to ExLlamaV3 extension sources..."
"$PYTHON" "$SCRIPT_DIR/patch_exllamav3_aarch64.py" "$EXLLAMAV3_SRC/exllamav3/exllamav3_ext"

# 3. Clean any stale JIT / build cache
rm -rf ~/.cache/torch_extensions/*exllama* /tmp/torch_extensions/*exllama* 2>/dev/null || true

# 4. Build and install into venv
echo "Compiling and installing exllamav3 into venv..."
# Uninstall first. The version in the source tree rarely changes between runs, so pip treats
# the requirement as already satisfied and skips the build entirely: edits to the ARM64 stub
# then appear to install successfully while the old extension is still in place.
"$PYTHON" -m pip uninstall -y exllamav3 >/dev/null 2>&1 || true
rm -f "$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"/exllamav3_ext*.so
"$PYTHON" -m pip install --no-build-isolation "$EXLLAMAV3_SRC"

# 4b. Name any unresolved symbol before the import does it cryptically. The ARM64 stub
#     replaces cpu/moe_mul1.cpp wholesale, so an entry upstream adds there links fine and
#     only fails at dlopen as "undefined symbol: _Z24exl3_moe_cpu_pool_stressiiii".
SO="$("$PYTHON" -c 'import sysconfig,glob,os; p=sysconfig.get_paths()["purelib"]; g=glob.glob(os.path.join(p,"exllamav3_ext*.so")); print(g[0] if g else "")')"
if [[ -n "$SO" ]] && command -v nm >/dev/null 2>&1; then
  MISSING="$(nm -D --undefined-only "$SO" 2>/dev/null | awk '{print $NF}' | grep -E 'exl3_|moe_cpu' || true)"
  if [[ -n "$MISSING" ]]; then
    echo "ERROR: the built extension has unresolved ExLlamaV3 symbols:" >&2
    echo "$MISSING" | while read -r s; do echo "  $(c++filt "$s" 2>/dev/null || echo "$s")" >&2; done
    echo "Add a stub for each to scripts/patch_exllamav3_aarch64.py, matching cpu/moe_mul1.h." >&2
    exit 1
  fi
  echo "Symbol check: no unresolved ExLlamaV3 symbols in $(basename "$SO")"
fi

# 5. Verify import and symbol resolution
echo "Verifying exllamav3 and exllamav3_ext import..."
"$PYTHON" - <<'PY'
import torch
import exllamav3
import exllamav3_ext
print(f"SUCCESS: exllamav3 imported from {exllamav3.__file__}")
print(f"SUCCESS: exllamav3_ext loaded from {exllamav3_ext.__file__}")
print(f"Device: {torch.cuda.get_device_name(0)} (capability {torch.cuda.get_device_capability(0)})")
PY

echo "=== ExLlamaV3 ARM64 Build and Verification Complete! ==="
