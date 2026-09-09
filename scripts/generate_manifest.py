#!/usr/bin/env python3
"""Environment and Model Manifest Generator for NVIDIA DGX Spark GB10 sm_121."""

import os
import sys
import json
import hashlib
import subprocess
from typing import Dict, Any

def get_git_info(repo_dir: str) -> Dict[str, Any]:
    info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, text=True).strip()
        info["commit"] = commit
        info["branch"] = branch
        info["dirty"] = len(status) > 0
    except Exception as e:
        info["error"] = str(e)
    return info

def get_package_versions() -> Dict[str, str]:
    packages = ["torch", "sglang", "flashinfer", "triton", "safetensors"]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed (no __version__)")
        except ImportError:
            versions[pkg] = "not installed"
        except Exception as e:
            versions[pkg] = f"error: {e}"
    
    # Check exllamav3
    try:
        import exllamav3_ext
        versions["exllamav3_ext"] = "available"
    except ImportError:
        versions["exllamav3_ext"] = "not available"

    try:
        import sglang_exl3_c
        versions["sglang_exl3_c"] = "available"
    except ImportError:
        versions["sglang_exl3_c"] = "not available"

    return versions

def get_memory_info() -> Dict[str, Any]:
    mem_info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    if key in ["MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached"]:
                        val_kb = int(parts[1].strip().split()[0])
                        mem_info[key] = f"{val_kb / (1024 * 1024):.2f} GiB"
    except Exception as e:
        mem_info["error"] = str(e)
    return mem_info

def get_model_info(model_path: str) -> Dict[str, Any]:
    info = {"path": model_path, "config_found": False}
    config_file = os.path.join(model_path, "config.json")
    if os.path.isfile(config_file):
        info["config_found"] = True
        try:
            with open(config_file, "rb") as f:
                content = f.read()
                info["config_sha256"] = hashlib.sha256(content).hexdigest()[:16]
            cfg = json.loads(content.decode("utf-8"))
            info["num_hidden_layers"] = cfg.get("num_hidden_layers")
            info["hidden_size"] = cfg.get("hidden_size")
            info["n_routed_experts"] = cfg.get("n_routed_experts")
            info["num_experts_per_tok"] = cfg.get("num_experts_per_tok")
            info["swiglu_limit"] = cfg.get("swiglu_limit")
            info["dspark_block_size"] = cfg.get("dspark_block_size")
            info["quantization_config"] = cfg.get("quantization_config", {})
        except Exception as e:
            info["error"] = str(e)
    return info

def generate_manifest(model_path: str = "/home/victor/models/DSV4-Flash-Vision-ablit-EXL3-MixedK") -> Dict[str, Any]:
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    manifest = {
        "hardware": "NVIDIA DGX Spark GB10 (sm_121 Blackwell), 128 GB unified system memory",
        "git": get_git_info(repo_dir),
        "packages": get_package_versions(),
        "memory": get_memory_info(),
        "model": get_model_info(model_path),
    }

    # CUDA Info
    try:
        import torch
        if torch.cuda.is_available():
            manifest["cuda"] = {
                "device_name": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
                "device_count": torch.cuda.device_count(),
                "torch_cuda_version": torch.version.cuda,
            }
    except Exception as e:
        manifest["cuda"] = {"error": str(e)}

    return manifest

if __name__ == "__main__":
    model_p = sys.argv[1] if len(sys.argv) > 1 else "/home/victor/models/DSV4-Flash-Vision-ablit-EXL3-MixedK"
    manifest = generate_manifest(model_p)
    print(json.dumps(manifest, indent=2))
