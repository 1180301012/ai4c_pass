"""
Fused pass for: softmax(in_1, dim=1) -> reshape -> view -> view -> * in_0 -> sum(dim=1) -> contiguous
Handles batch_size = 1 graphs.

in_1 shape: [1, 2, 1, H]  (softmax over 2 groups)
in_0 shape: [1, 2, H, C, W]
output:     [1, H, C, W]

Uses shared dispatch routing to avoid replacement_func_limit.
"""

import torch
import triton
import triton.language as tl


# ─── Shared Triton kernel (same logic, dispatched by batch route) ─────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64},   num_warps=2),
        triton.Config({'BLOCK_W': 128},  num_warps=4),
        triton.Config({'BLOCK_W': 256},  num_warps=4),
        triton.Config({'BLOCK_W': 512},  num_warps=8),
        triton.Config({'BLOCK_W': 1024}, num_warps=8),
    ],
    key=['H', 'C', 'W'],
)
@triton.jit
def _fused_ws_kernel(
    w_ptr,    # [B, 2, 1, H] – w[b, j, 0, h] = w_ptr + b*(2*H) + j*H + h
    in_ptr,   # [B, 2, H, C, W]
    out_ptr,  # [B, H, C, W]
    H, C, W,
    BLOCK_W: tl.constexpr,
):
    pid   = tl.program_id(0)
    c_idx = pid % C
    h_idx = (pid // C) % H
    b_idx = pid // (H * C)

    # Load softmax weights for this (b, h)
    w0 = tl.load(w_ptr + b_idx * (2 * H) + h_idx).to(tl.float32)
    w1 = tl.load(w_ptr + b_idx * (2 * H) + H + h_idx).to(tl.float32)

    # Numerically stable softmax over 2 values
    max_w  = tl.maximum(w0, w1)
    e0     = tl.exp(w0 - max_w)
    e1     = tl.exp(w1 - max_w)
    inv_s  = 1.0 / (e0 + e1)
    s0     = e0 * inv_s
    s1     = e1 * inv_s

    in_base  = b_idx * (2 * H * C * W) + h_idx * (C * W)
    out_base = b_idx * (H * C * W) + h_idx * (C * W)

    for w_start in tl.range(0, W, BLOCK_W):
        w_offsets = w_start + tl.arange(0, BLOCK_W)
        mask      = w_offsets < W

        in0_off = in_base + 0 * (C * W) + c_idx * W + w_offsets
        in1_off = in_base + 1 * (C * W) + c_idx * W + w_offsets

        v0 = tl.load(in_ptr + in0_off, mask=mask, other=0.0).to(tl.float32)
        v1 = tl.load(in_ptr + in1_off, mask=mask, other=0.0).to(tl.float32)

        result = s0 * v0 + s1 * v1

        out_off = out_base + c_idx * W + w_offsets
        tl.store(out_ptr + out_off, result, mask=mask)


def _run_ws_kernel(in_0, in_1):
    B, H = in_1.shape[0], in_1.shape[3]
    C    = in_0.shape[3]
    W    = in_0.shape[4]
    out  = torch.empty((B, H, C, W), dtype=in_0.dtype, device=in_0.device)
    _fused_ws_kernel[(B * H * C,)](in_1, in_0, out, H, C, W)
    return out


# ─── Shared dispatch wrapper (routing by batch) ───────────────────────────────

@torch.fx.wrap
def _fused_ws_dispatch(in_0, in_1, route):
    if route == "b1":
        return _run_ws_kernel(in_0, in_1)
    elif route == "b2":
        return _run_ws_kernel(in_0, in_1)
    else:  # route == "b8"
        return _run_ws_kernel(in_0, in_1)


# ─── Pattern / replacement API ────────────────────────────────────────────────

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(1, -1)
    tmp_2 = tmp_1.view(1, -1, 1, 1)
    tmp_3 = tmp_2.view(1, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return (tmp_6,)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "b1")


def replacement_func():
    return _fused_ws_dispatch