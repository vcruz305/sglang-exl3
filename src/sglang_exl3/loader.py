"""EXL3 shard loader for SGLang on NVIDIA DGX Spark GB10 (128 GiB unified memory).

Streams safetensors shards with pread() one tensor at a time and drops each
file range from the page cache after reading, so host memory peaks at about
1.0x the weight bytes (not 2x). Loading is sequential by design; see
``exl3_mmap_weights_iterator``.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import sys
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

logger = logging.getLogger("sglang.exl3.loader")

# Constant identifier for the custom EXL3 mmap loader
LOAD_FORMAT_EXL3_MMAP = "exl3_mmap"


class LoadFormatEXL3:
    EXL3_MMAP = LOAD_FORMAT_EXL3_MMAP


def is_unified_memory_platform() -> bool:
    """Detect whether host hardware provides unified CPU-GPU coherent memory (e.g. GB10 / GB200)."""
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        # Blackwell GB10 / GB200 sm_120 / sm_121 or high capability unified devices
        if props.major >= 10:
            return True
        # Check architecture or device name
        name = props.name.lower()
        if any(arch in name for arch in ("gb10", "gb200", "blackwell", "grace", "tegra", "spark")):
            return True
    except Exception:
        pass
    return False


def madvise_file_preload(file_path: str) -> bool:
    """Issue POSIX_MADV_WILLNEED on file pages to initiate kernel readahead."""
    if not hasattr(os, "posix_fadvise"):
        return False
    try:
        with open(file_path, "rb") as f:
            fd = f.fileno()
            total_size = os.fstat(fd).st_size
            POSIX_FADV_WILLNEED = getattr(os, "POSIX_FADV_WILLNEED", 3)
            os.posix_fadvise(fd, 0, total_size, POSIX_FADV_WILLNEED)
        return True
    except Exception:
        return False


_SAFETENSORS_DTYPES: Dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    _SAFETENSORS_DTYPES["F8_E4M3"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _SAFETENSORS_DTYPES["F8_E5M2"] = torch.float8_e5m2

_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)


def _drop_file_pages(fd: int, offset: int = 0, length: int = 0) -> None:
    """Best-effort: drop clean page-cache pages for a file range."""
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, offset, length, _FADV_DONTNEED)
        except OSError:
            pass


def host_memory_summary(device: Union[str, torch.device, None] = None) -> str:
    """One-line host/GPU memory snapshot for load-time logging."""
    mem_avail_gib = float("nan")
    swap_used_mib = float("nan")
    try:
        fields: Dict[str, int] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemAvailable", "SwapTotal", "SwapFree"):
                    fields[key] = int(rest.split()[0])  # kB
        mem_avail_gib = fields["MemAvailable"] / (1024 ** 2)
        swap_used_mib = (fields["SwapTotal"] - fields["SwapFree"]) / 1024
    except (OSError, KeyError, ValueError):
        pass
    cuda_gib = float("nan")
    try:
        if torch.cuda.is_available():
            cuda_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return (
        f"MemAvailable={mem_avail_gib:.2f}GiB swap_used={swap_used_mib:.0f}MiB "
        f"cuda_allocated={cuda_gib:.2f}GiB"
    )


def iter_safetensors_pread(path: str) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Stream one safetensors shard tensor-by-tensor with pread() (no mmap).

    Why: an mmap'd shard keeps its file pages in the page cache while the CUDA
    copies accumulate. On GB10 unified memory both count against the same 128 GiB,
    so the load cost ~2x the weight bytes (measured: page cache grew 1:1 with the
    bytes read, and it was not released before the next shard).

    Each tensor gets a fresh buffer that is freed once the consumer drops it, and
    the file range is dropped from the page cache right after it is read.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        (header_len,) = struct.unpack("<Q", os.pread(fd, 8, 0))
        header = json.loads(os.pread(fd, header_len, 8))
        header.pop("__metadata__", None)
        base = 8 + header_len
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            dtype_name = info["dtype"]
            if dtype_name not in _SAFETENSORS_DTYPES:
                raise RuntimeError(
                    f"[EXL3 MMap Loader] unsupported safetensors dtype {dtype_name} for {name} in {path}"
                )
            dtype = _SAFETENSORS_DTYPES[dtype_name]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            nbytes = end - start
            if nbytes == 0:
                yield name, torch.empty(shape, dtype=dtype)
                continue
            offset = base + start
            buf = bytearray(nbytes)
            view = memoryview(buf)
            got = 0
            while got < nbytes:
                n = os.preadv(fd, [view[got:]], offset + got)
                if n <= 0:
                    raise IOError(f"[EXL3 MMap Loader] short read in {path} at {name}: {got}/{nbytes} bytes")
                got += n
            view.release()
            _drop_file_pages(fd, offset, nbytes)
            tensor = torch.frombuffer(buf, dtype=dtype).reshape(shape)
            del buf
            yield name, tensor
            del tensor
    finally:
        _drop_file_pages(fd)
        os.close(fd)


def exl3_mmap_weights_iterator(
    hf_weights_files: List[str],
    device: Union[str, torch.device] = "cpu",
    num_threads: int = 8,
    prefetch_shards: int = 2,
    drop_cache_after_load: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over safetensors weights without keeping shard mmaps or page cache.

    Shards are read sequentially with pread() (see ``iter_safetensors_pread``).
    ``num_threads``, ``prefetch_shards`` and ``drop_cache_after_load`` are accepted
    for API compatibility; the page cache is always released after each read,
    because parallel readers and a resident page cache were the cause of the
    ~2x host-memory peak on unified-memory GB10.

    Args:
        hf_weights_files: Paths to safetensors checkpoint shards, in load order.
        device: Target device the consumer copies into (informational).
    """
    total_files = len(hf_weights_files)
    if total_files == 0:
        return

    unified_mem = is_unified_memory_platform()
    total_bytes = 0
    total_tensors = 0
    start_time = time.perf_counter()

    enable_tqdm = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    pbar = tqdm(
        total=total_files,
        desc=f"EXL3 MMap Loading ({'Unified sm_121' if unified_mem else 'Zero-Copy'})",
        disable=not enable_tqdm,
    )
    try:
        for index, filepath in enumerate(hf_weights_files, start=1):
            shard_tensors = 0
            for key, tensor in iter_safetensors_pread(filepath):
                total_bytes += tensor.nbytes
                total_tensors += 1
                shard_tensors += 1
                yield key, tensor
            pbar.update(1)
            logger.info(
                "[EXL3 MMap Loader] shard %d/%d %s tensors=%d %s",
                index, total_files, os.path.basename(filepath), shard_tensors,
                host_memory_summary(device),
            )
    finally:
        pbar.close()
        elapsed = time.perf_counter() - start_time
        total_gb = total_bytes / (1024 ** 3)
        throughput = (total_gb / elapsed) if elapsed > 0 else 0.0
        logger.info(
            f"[EXL3 MMap Loader] Successfully read {total_tensors} tensors "
            f"({total_gb:.2f} GiB) across {total_files} shards in {elapsed:.2f}s "
            f"({throughput:.2f} GB/s effective throughput). {host_memory_summary(device)}"
        )

try:
    from sglang.srt.model_loader.loader import DefaultModelLoader, LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.configs.device_config import DeviceConfig
    _HAS_SGLANG_LOADER = True
except ImportError:
    _HAS_SGLANG_LOADER = False
    DefaultModelLoader = object  # type: ignore
    LoadConfig = object  # type: ignore
    ModelConfig = object  # type: ignore
    DeviceConfig = object  # type: ignore


class EXL3MMapModelLoader(DefaultModelLoader):
    """High-performance zero-copy unified memory model loader for SGLang on GB10 / Blackwell."""

    def __init__(self, load_config: Any):
        if _HAS_SGLANG_LOADER:
            super().__init__(load_config)
        else:
            self.load_config = load_config
        self.load_format = LOAD_FORMAT_EXL3_MMAP

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Tuple[str, List[str], bool]:
        """Resolve local/HF safetensor shards while retaining the mmap format marker.

        SGLang's base implementation only knows its built-in ``LoadFormat``
        enum and rejects ``exl3_mmap``.  Temporarily present this custom loader
        as ``safetensors`` for shard discovery, then restore the custom value so
        downstream dispatch and diagnostics continue to identify EXL3 mmap.
        """
        if not _HAS_SGLANG_LOADER:
            return super()._prepare_weights(model_name_or_path, revision, fall_back_to_pt)

        from sglang.srt.configs.load_config import LoadFormat

        original_format = self.load_config.load_format
        self.load_config.load_format = LoadFormat.SAFETENSORS
        try:
            return super()._prepare_weights(model_name_or_path, revision, fall_back_to_pt)
        finally:
            self.load_config.load_format = original_format

    def _get_weights_iterator(
        self,
        source: Any,
        *,
        resolved_source: Optional[Any] = None,
        startup_prefetch_started: bool = False,
        startup_prefetch_active: bool = False,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Override weights iterator to use parallel zero-copy mmap."""
        extra_config = getattr(self.load_config, "model_loader_extra_config", {}) or {}
        num_threads = extra_config.get("num_threads", 8)
        prefetch_shards = extra_config.get("prefetch_shards", 2)
        drop_cache = extra_config.get("drop_cache_after_load", False)

        if resolved_source is None:
            hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
                source.model_or_path, source.revision, source.fall_back_to_pt
            )
        else:
            hf_weights_files = list(resolved_source.weight_files)
            use_safetensors = resolved_source.use_safetensors

        if not use_safetensors:
            return super()._get_weights_iterator(
                source,
                resolved_source=resolved_source,
                startup_prefetch_started=startup_prefetch_started,
                startup_prefetch_active=startup_prefetch_active,
            )

        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        weights_iterator = exl3_mmap_weights_iterator(
            hf_weights_files=hf_weights_files,
            device=target_device,
            num_threads=num_threads,
            prefetch_shards=prefetch_shards,
            drop_cache_after_load=drop_cache,
        )

        prefix = getattr(source, "prefix", "")
        return ((prefix + name, tensor) for (name, tensor) in weights_iterator)
