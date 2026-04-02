"""
Fused pass: view + softmax + view
Targets shape [8, 625, 625] (float32 variant).

Minimal single-input pattern: match just the view+softmax+view part.
The add before and dropout after are left in the remaining graph.
in_x receives the already-added [1,8,625,625] tensor.
"""

import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_x):
    tmp_1 = torch.ops.aten.view.default(in_x, [8, 625, 625])
    tmp_2 = torch.ops.aten._softmax.default(tmp_1, -1, False)
    tmp_3 = torch.ops.aten.view.default(tmp_2, [1, 8, 625, 625])
    return tmp_3


def replacement_args(in_x):
    return (in_x,)


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 1024}, num_warps=2, num_stages=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def _softmax_625_625_kernel(
    in_ptr,    # [1, 8, H, W] – input (4-D)
    out_ptr,   # [8, H, W]    – output (3-D)
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row_idx = tl.program_id(0)
    head    = row_idx // H
    h       = row_idx %  H

    in_off  = head * H * W + h * W
    out_off = row_idx * W

    cols = tl.arange(0, BLOCK_W)
    mask = cols < W

    x_raw = tl.load(in_ptr + in_off + cols, mask=mask, other=0.0)
    x     = x_raw.to(tl.float32)

    x_safe = tl.where(mask, x, float('-inf'))
    x_max  = tl.max(x_safe, axis=0)
    x_exp  = tl.exp(x - x_max)
    x_exp  = tl.where(mask, x_exp, 0.0)
    x_sum  = tl.sum(x_exp, axis=0)
    out_f32 = x_exp / x_sum

    tl.store(out_ptr + out_off + cols, out_f32.to(x_raw.dtype), mask=mask)


# ── Replacement wrapper ───────────────────────────────────────────────────────

@torch.fx.wrap
def fused_softmax_8_625_625(in_x):
    """
    in_x : [1, 8, 625, 625] – the add result
    returns : [1, 8, 625, 625] – softmax output (= tmp_3)
    """
    _HEADS = 8
    _H     = 625
    _W     = 625

    out = torch.empty(_HEADS, _H, _W, dtype=in_x.dtype, device=in_x.device)

    N_rows = _HEADS * _H   # 5000
    _softmax_625_625_kernel[(N_rows,)](
        in_x, out,
        H=_H, W=_W,
    )

    return out.view(1, _HEADS, _H, _W)


def replacement_func():
    return fused_softmax_8_625_625