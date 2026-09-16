from pathlib import Path
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent
ext_include = os.environ.get("EXL3_EXT_INCLUDE")
if ext_include:
    ext_include = Path(ext_include)
else:
    try:
        import exllamav3
        ext_include = Path(exllamav3.__file__).resolve().parent / "exllamav3_ext"
    except Exception:
        ext_include = None

include_dirs = [str(ROOT / "csrc")]
if ext_include and ext_include.is_dir():
    include_dirs.append(str(ext_include))
    if (ext_include / "quant").is_dir():
        include_dirs.append(str(ext_include / "quant"))

_ext_modules = []
_cmdclass = {}
if CUDA_HOME:
    _ext_modules = [
        CUDAExtension(
            name="sglang_exl3_c",
            sources=[
                "csrc/bindings.cpp",
                "csrc/exl3_gemv.cu",
                "csrc/p2b_batched.cu",
                "csrc/p2b_moe.cu",
                "csrc/exl3_gemm.cu",
                "csrc/exl3_fat_gemm.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                # Torch 2.14+ headers (ExpandUtils / autograd) need C++20.
                # c++17 fails to parse at::symint::sizes on GB10 / cu130.
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20", "-gencode=arch=compute_121a,code=sm_121a"],
            },
        )
    ]
    _cmdclass = {"build_ext": BuildExtension}

setup(
    name="sglang-exl3",
    version="0.2.1",
    description="Native sm_121 Blackwell EXL3 quantization plugin for SGLang with Dynamic FP8 DSpark stages and a compiled batched MoE kernel",
    author="Victor Cruz",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=_ext_modules,
    entry_points={
        "sglang.srt.plugins": ["exl3=sglang_exl3:register"],
    },
    cmdclass=_cmdclass,
)
