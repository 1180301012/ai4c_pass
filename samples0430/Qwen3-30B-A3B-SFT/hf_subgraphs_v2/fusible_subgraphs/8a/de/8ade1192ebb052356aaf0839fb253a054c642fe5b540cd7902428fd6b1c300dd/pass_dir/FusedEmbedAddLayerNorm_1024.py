import torch
import triton
import triton.language as tl
from torch import device as torch_device


# ── Kernels (unique per file) ─────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _embed_ln_k1024(x_ptr, pos_ptr, w_ptr, b_ptr, out_ptr, in4_ptr,
                    seq_len, D: tl.constexpr, BLOCK_D: tl.constexpr, eps):
    s = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    idx = tl.load(in4_ptr + s) + 2
    x = tl.load(x_ptr + s * D + offs, mask=mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + idx * D + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + pos
    mean = tl.sum(added, axis=0) / D
    diff = tl.where(mask, added - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = tl.where(mask, (added - mean) * rstd, 0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + s * D + offs, normed * w + b, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _embed_ln_k768(x_ptr, pos_ptr, w_ptr, b_ptr, out_ptr, in4_ptr,
                   seq_len, D: tl.constexpr, BLOCK_D: tl.constexpr, eps):
    s = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    idx = tl.load(in4_ptr + s) + 2
    x = tl.load(x_ptr + s * D + offs, mask=mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + idx * D + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + pos
    mean = tl.sum(added, axis=0) / D
    diff = tl.where(mask, added - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = tl.where(mask, (added - mean) * rstd, 0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + s * D + offs, normed * w + b, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _embed_ln_k16(x_ptr, pos_ptr, w_ptr, b_ptr, out_ptr, in4_ptr,
                  seq_len, D: tl.constexpr, BLOCK_D: tl.constexpr, eps):
    s = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    idx = tl.load(in4_ptr + s) + 2
    x = tl.load(x_ptr + s * D + offs, mask=mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + idx * D + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + pos
    mean = tl.sum(added, axis=0) / D
    diff = tl.where(mask, added - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = tl.where(mask, (added - mean) * rstd, 0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + s * D + offs, normed * w + b, mask=mask)


# ── Shared dispatch wrapper ───────────────────────────────────────────────────
# All three pass files import _fused_embed_ln_dispatch from _kernels.py so
# replacement_func() returns the EXACT SAME function object everywhere.
from pass_dir._kernels import _fused_embed_ln_dispatch  # noqa: F401


# ── Pattern & plumbing ────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(torch_device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)
    return tmp_14


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return _fused_embed_ln_dispatch