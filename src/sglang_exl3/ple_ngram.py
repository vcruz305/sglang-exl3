"""EXL3 n-gram PLE gather for SGLang. Stream packed rows; never materialize 95GiB BF16."""
from __future__ import annotations

import json
import mmap
import os
import re
import struct
from pathlib import Path

import torch

ROW_DIM = 160
MUL1 = 0x83DCD12D
_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.trellis$")


def words_per_row(K: int) -> int:
    return 1 + ROW_DIM * K // 16


def mul1_codebook(device) -> torch.Tensor:
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    h = (1024 + bsum).float()
    k_inv = torch.tensor([0x1EEE], dtype=torch.uint16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xC931], dtype=torch.uint16).view(torch.float16).float().item()
    return (h * k_inv + k_bias).to(torch.float16)


def unpack_rows(packed: torch.Tensor, K: int):
    dev = packed.device
    scales = packed[:, 0].contiguous().view(torch.float16)
    words = packed[:, 1:].view(torch.uint16).to(torch.int64)
    stream = ((words.unsqueeze(-1) >> torch.arange(16, device=dev)) & 1).reshape(
        packed.shape[0], ROW_DIM * K
    )
    i = torch.arange(ROW_DIM, device=dev).unsqueeze(1)
    m = torch.arange(16, device=dev).unsqueeze(0)
    src = ((i - m // K) % ROW_DIM) * K + m % K
    states = (stream[:, src] << m).sum(dim=-1)
    return states, scales


def dequant_rows(packed, K, codebook, bias=None):
    states, scales = unpack_rows(packed, K)
    out = codebook[states].float() * scales.float().unsqueeze(1)
    if bias is not None:
        out = out + bias.float()
    return out


class EXL3NgramTable:
    def __init__(self, path: str):
        self.path = path
        self.fh = open(path, "rb")
        self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack_from("<Q", self.mm, 0)[0]
        header = json.loads(self.mm[8 : 8 + header_len])
        self.data0 = 8 + header_len
        shards = []
        aux = {}
        for k, meta in header.items():
            if k == "__metadata__":
                continue
            off0, off1 = meta["data_offsets"]
            abs0 = self.data0 + off0
            shape = tuple(meta["shape"])
            dtype = meta["dtype"]
            rec = (abs0, shape, dtype)
            m = _SHARD_RE.search(k)
            if m:
                shards.append((int(m.group(1)), rec))
            else:
                aux[k.rsplit(".", 1)[-1]] = rec
        shards.sort()
        self.shard_off = [rec[0] for _, rec in shards]
        self.rows_per_shard = shards[0][1][1][0]
        self.words = shards[0][1][1][1]
        self.K = (self.words - 1) * 16 // ROW_DIM
        self.row_bytes = self.words * 2
        self.num_rows = sum(rec[1][0] for _, rec in shards)
        self._aux_meta = aux
        self.head_bias = self._read_tensor("head_bias").to(torch.float16)
        self.head_offsets = self._read_tensor("head_offsets").long()
        self.codebook = None
        print(
            f"[sglang_exl3] PLE ngram mmap {path} shards={len(shards)} "
            f"rows={self.num_rows} K={self.K} row_bytes={self.row_bytes}",
            flush=True,
        )

    def _read_tensor(self, suffix: str) -> torch.Tensor:
        abs0, shape, dtype = self._aux_meta[suffix]
        n = 1
        for s in shape:
            n *= s
        buf = self.mm[abs0 : abs0 + n * _st_nbytes(dtype)]
        t = torch.frombuffer(bytearray(buf), dtype=_st_dtype(dtype)).clone()
        return t.reshape(shape)

    def fetch_packed(self, uids: torch.Tensor) -> torch.Tensor:
        ids = uids.detach().to("cpu", torch.int64).reshape(-1)
        n = ids.numel()
        out = torch.empty((n, self.words), dtype=torch.int16)
        rps = self.rows_per_shard
        rb = self.row_bytes
        mm = self.mm
        offs = self.shard_off
        last = len(offs) - 1
        for i, uid in enumerate(ids.tolist()):
            uid = int(uid)
            if uid < 0:
                out[i].zero_()
                continue
            s = min(uid // rps, last)
            local = uid - s * rps
            if s == last:
                # last shard may be short; clamp
                max_local = (self.num_rows - s * rps) - 1
                if local > max_local:
                    out[i].zero_()
                    continue
            abs_off = offs[s] + local * rb
            out[i] = torch.frombuffer(
                bytearray(mm[abs_off : abs_off + rb]), dtype=torch.int16
            )
        return out

    def gather(self, emb, input_ids: torch.Tensor, out=None) -> torch.Tensor:
        expected = (*input_ids.shape, 160)
        if out is None:
            output = torch.empty(expected, dtype=torch.bfloat16, device=input_ids.device)
        else:
            output = out
        flat = input_ids.reshape(-1)
        if flat.numel() == 0:
            return output
        packed = self.fetch_packed(flat)
        device = input_ids.device
        if self.codebook is None or self.codebook.device != device:
            self.codebook = mul1_codebook(device)
        packed_d = packed.to(device, non_blocking=True)
        uids_d = flat.to(device, torch.int64)
        heads = (
            torch.searchsorted(self.head_offsets.to(device), uids_d, right=True) - 1
        ).clamp(0, self.head_bias.shape[0] - 1)
        bias = self.head_bias.to(device)[heads]
        rows = dequant_rows(packed_d, self.K, self.codebook, bias)
        output.view(-1, 160).copy_(rows.to(torch.bfloat16))
        return output


def _st_dtype(s: str):
    return {
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U16": torch.int16,
        "U32": torch.int32,
        "U64": torch.int64,
    }[s]


def _st_nbytes(s: str) -> int:
    return {
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "I16": 2,
        "I32": 4,
        "I64": 8,
        "U16": 2,
        "U32": 4,
        "U64": 8,
    }[s]


_TABLE = None


def ngram_path() -> str:
    p = os.environ.get("SGLANG_EXL3_NGRAM", "")
    if p and Path(p).is_file():
        return p
    cand = Path("/root/models/Qwen3.8-Flash-Next-exl3-3.05bpw/ngram_embedding.safetensors")
    if cand.is_file():
        return str(cand)
    raise FileNotFoundError("SGLANG_EXL3_NGRAM not set and default ngram file missing")


def get_table() -> EXL3NgramTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = EXL3NgramTable(ngram_path())
    return _TABLE


def consume_exl3_ngram_tensor(name: str, loaded_weight, ple_modules: dict) -> bool:
    """Return True if this checkpoint key is EXL3 ngram (consumed / skip warn)."""
    if ".ngram_embedding." not in name:
        return False
    suffix = name.rsplit(".", 1)[-1]
    if suffix in ("trellis", "head_bias", "head_offsets", "head_vocab_sizes", "layer_multipliers"):
        table = get_table()
        # attach gather override once
        for mod in ple_modules.values():
            emb = getattr(mod, "ngram_embedding", None)
            if emb is None:
                continue
            if getattr(emb, "_exl3_ngram", None) is None:
                emb._exl3_ngram = table
                _orig = emb.gather

                def _gather(self, input_ids, out=None, _orig=_orig):
                    return self._exl3_ngram.gather(self, input_ids, out)

                emb.gather = _gather.__get__(emb, type(emb))
            if suffix == "head_offsets" and hasattr(mod, "ngram_heads_offsets"):
                if tuple(mod.ngram_heads_offsets.shape) == tuple(loaded_weight.shape):
                    mod.ngram_heads_offsets.copy_(
                        loaded_weight.to(
                            device=mod.ngram_heads_offsets.device,
                            dtype=mod.ngram_heads_offsets.dtype,
                        )
                    )
            if suffix == "head_vocab_sizes" and hasattr(mod, "ngram_heads_vocab_sizes"):
                if tuple(mod.ngram_heads_vocab_sizes.shape) == tuple(loaded_weight.shape):
                    mod.ngram_heads_vocab_sizes.copy_(
                        loaded_weight.to(
                            device=mod.ngram_heads_vocab_sizes.device,
                            dtype=mod.ngram_heads_vocab_sizes.dtype,
                        )
                    )
            if suffix == "layer_multipliers" and hasattr(mod, "layer_multipliers"):
                if tuple(mod.layer_multipliers.shape) == tuple(loaded_weight.shape):
                    mod.layer_multipliers.copy_(
                        loaded_weight.to(
                            device=mod.layer_multipliers.device,
                            dtype=mod.layer_multipliers.dtype,
                        )
                    )
        return True
    return False
