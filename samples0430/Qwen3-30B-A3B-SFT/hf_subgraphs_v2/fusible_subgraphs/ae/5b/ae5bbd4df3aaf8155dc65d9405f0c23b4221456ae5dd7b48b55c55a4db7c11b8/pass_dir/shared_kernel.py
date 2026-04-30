"""
Shared Triton kernels and dispatch wrapper for WavLM relative-position fusion.

Supports 4 variants:
  - wavlm_base  (N_heads=12, dtype=bfloat16 or float16)
  - wavlm_large (N_heads=16, dtype=bfloat16 or float16)
  All variants: S=199, K=64

Each kernel uses a 2D grid (pid_s, pid_h) where:
  pid_s = sequence block index
  pid_h = head index (0..N_heads-1)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: N_heads=12, S=199, dtype=bfloat16
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_12_bf16(
    x_ptr, w_ptr, bias_ptr, in2_ptr, out_ptr,
    N_heads, K: tl.constexpr, S: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    h = tl.program_id(1)

    s_start = pid_s * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    k_offs = tl.arange(0, K)
    k_half = tl.arange(0, 4)

    # Load x[h, s_offs, :]  [BLOCK_S, K]
    x_base = h * S * K
    x = tl.load(
        x_ptr + x_base + s_offs[:, None] * K + k_offs[None, :],
        mask=s_mask[:, None], other=0.0
    ).to(tl.float32)

    # Load w[:4, :]  [4, K]  and  w[4:8, :]  [4, K]
    w0 = tl.load(w_ptr + tl.arange(0, 4)[:, None] * K + k_offs[None, :]).to(tl.float32)
    w1 = tl.load(w_ptr + tl.arange(4, 8)[:, None] * K + k_offs[None, :]).to(tl.float32)

    # Load bias[:4] and bias[4:8]
    bias0 = tl.load(bias_ptr + tl.arange(0, 4)).to(tl.float32)
    bias1 = tl.load(bias_ptr + tl.arange(4, 8)).to(tl.float32)

    # Compute dot products  [BLOCK_S, 4]
    d0 = tl.dot(x, tl.trans(w0)) + bias0[None, :]
    d1 = tl.dot(x, tl.trans(w1)) + bias1[None, :]

    # Sum over K dim → [BLOCK_S]
    sum0 = tl.sum(d0, axis=1)
    sum1 = tl.sum(d1, axis=1)

    # Sigmoid
    sig0 = tl.sigmoid(sum0)
    sig1 = tl.sigmoid(sum1)

    # Load in2[h, s_offs]  [BLOCK_S]
    in2 = tl.load(in2_ptr + h * S + s_offs, mask=s_mask, other=0.0).to(tl.float32)

    # out = sig0 * (sig1 * in2 - 1.0) + 2.0
    out_val = sig0 * (sig1 * in2 - 1.0) + 2.0

    # Store single value at out[0, h, s_offs, 0] — flat index = h*S + s
    out_offs = h * S + s_offs
    tl.store(out_ptr + out_offs, out_val.to(x_ptr.dtype.element_ty), mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel: N_heads=16, S=199, dtype=bfloat16
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_16_bf16(
    x_ptr, w_ptr, bias_ptr, in2_ptr, out_ptr,
    N_heads, K: tl.constexpr, S: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    h = tl.program_id(1)

    s_start = pid_s * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    k_offs = tl.arange(0, K)
    k_half = tl.arange(0, 4)

    x_base = h * S * K
    x = tl.load(
        x_ptr + x_base + s_offs[:, None] * K + k_offs[None, :],
        mask=s_mask[:, None], other=0.0
    ).to(tl.float32)

    w0 = tl.load(w_ptr + tl.arange(0, 4)[:, None] * K + k_offs[None, :]).to(tl.float32)
    w1 = tl.load(w_ptr + tl.arange(4, 8)[:, None] * K + k_offs[None, :]).to(tl.float32)

    bias0 = tl.load(bias_ptr + tl.arange(0, 4)).to(tl.float32)
    bias1 = tl.load(bias_ptr + tl.arange(4, 8)).to(tl.float32)

    d0 = tl.dot(x, tl.trans(w0)) + bias0[None, :]
    d1 = tl.dot(x, tl.trans(w1)) + bias1[None, :]

    sum0 = tl.sum(d0, axis=1)
    sum1 = tl.sum(d1, axis=1)

    sig0 = tl.sigmoid(sum0)
    sig1 = tl.sigmoid(sum1)

    in2 = tl.load(in2_ptr + h * S + s_offs, mask=s_mask, other=0.0).to(tl.float32)

    out_val = sig0 * (sig1 * in2 - 1.0) + 2.0

    # Store single value at out[0, h, s_offs, 0] — flat index = h*S + s
    out_offs = h * S + s_offs
    tl.store(out_ptr + out_offs, out_val.to(x_ptr.dtype.element_ty), mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel: N_heads=12, S=199, dtype=float16
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_12_fp16(
    x_ptr, w_ptr, bias_ptr, in2_ptr, out_ptr,
    N_heads, K: tl.constexpr, S: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    h = tl.program_id(1)

    s_start = pid_s * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    k_offs = tl.arange(0, K)

    x_base = h * S * K
    x = tl.load(
        x_ptr + x_base + s_offs[:, None] * K + k_offs[None, :],
        mask=s_mask[:, None], other=0.0
    ).to(tl.float32)

    w0 = tl.load(w_ptr + tl.arange(0, 4)[:, None] * K + k_offs[None, :]).to(tl.float32)
    w1 = tl.load(w_ptr + tl.arange(4, 8)[:, None] * K + k_offs[None, :]).to(tl.float32)

    bias0 = tl.load(bias_ptr + tl.arange(0, 4)).to(tl.float32)
    bias1 = tl.load(bias_ptr + tl.arange(4, 8)).to(tl.float32)

    d0 = tl.dot(x, tl.trans(w0)) + bias0[None, :]
    d1 = tl.dot(x, tl.trans(w1)) + bias1[None, :]

    sum0 = tl.sum(d0, axis=1)
    sum1 = tl.sum(d1, axis=1)

    sig0 = tl.sigmoid(sum0)
    sig1 = tl.sigmoid(sum1)

    in2 = tl.load(in2_ptr + h * S + s_offs, mask=s_mask, other=0.0).to(tl.float32)

    out_val = sig0 * (sig1 * in2 - 1.0) + 2.0

    # Store single value at out[0, h, s_offs, 0] — flat index = h*S + s
    out_offs = h * S + s_offs
    tl.store(out_ptr + out_offs, out_val.to(x_ptr.dtype.element_ty), mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel: N_heads=16, S=199, dtype=float16
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_16_fp16(
    x_ptr, w_ptr, bias_ptr, in2_ptr, out_ptr,
    N_heads, K: tl.constexpr, S: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    h = tl.program_id(1)

    s_start = pid_s * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    k_offs = tl.arange(0, K)

    x_base = h * S * K
    x = tl.load(
        x_ptr + x_base + s_offs[:, None] * K + k_offs[None, :],
        mask=s_mask[:, None], other=0.0
    ).to(tl.float32)

    w0 = tl.load(w_ptr + tl.arange(0, 4)[:, None] * K + k_offs[None, :]).to(tl.float32)
    w1 = tl.load(w_ptr + tl.arange(4, 8)[:, None] * K + k_offs[None, :]).to(tl.float32)

    bias0 = tl.load(bias_ptr + tl.arange(0, 4)).to(tl.float32)
    bias1 = tl.load(bias_ptr + tl.arange(4, 8)).to(tl.float32)

    d0 = tl.dot(x, tl.trans(w0)) + bias0[None, :]
    d1 = tl.dot(x, tl.trans(w1)) + bias1[None, :]

    sum0 = tl.sum(d0, axis=1)
    sum1 = tl.sum(d1, axis=1)

    sig0 = tl.sigmoid(sum0)
    sig1 = tl.sigmoid(sum1)

    in2 = tl.load(in2_ptr + h * S + s_offs, mask=s_mask, other=0.0).to(tl.float32)

    out_base = h * S * 2
    out_offs = out_base + s_offs * 2
    tl.store(out_ptr + out_offs, out_val.to(x_ptr.dtype.element_ty), mask=s_mask)
    tl.store(out_ptr + out_offs + 1, out_val.to(x_ptr.dtype.element_ty), mask=s_mask)


def _run_12(x, weight, bias, in2):
    """N_heads=12, S=199, K=64, BLOCK_S=32."""
    S, K, N_heads, BLOCK_S = 199, 64, 12, 32
    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S
    # Allocate output directly in the expected final shape [1, N, S, 1]
    out = torch.empty((1, N_heads, S, 1), dtype=x.dtype, device=x.device)
    _kernel_12_bf16[(num_s_blocks, N_heads)](
        x, weight, bias, in2, out,
        N_heads=N_heads, K=K, S=S, BLOCK_S=BLOCK_S, num_warps=4,
    )
    return out


def _run_16(x, weight, bias, in2):
    """N_heads=16, S=199, K=64, BLOCK_S=32."""
    S, K, N_heads, BLOCK_S = 199, 64, 16, 32
    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S
    # Allocate output directly in the expected final shape [1, N, S, 1]
    out = torch.empty((1, N_heads, S, 1), dtype=x.dtype, device=x.device)
    _kernel_16_bf16[(num_s_blocks, N_heads)](
        x, weight, bias, in2, out,
        N_heads=N_heads, K=K, S=S, BLOCK_S=BLOCK_S, num_warps=4,
    )
    return out


@torch.fx.wrap
def wavlm_linear_dispatch(in_0, in_1, in_2, in_3, route):
    """
    Dispatch wrapper shared across all 4 pass files.
    in_0: bias   [8]
    in_1: weight [8, 64]
    in_2: in2    [1, N, 1, 1]
    in_3: x      [1, N, 199, 64]
    route: "12_bf16" | "16_bf16" | "12_fp16" | "16_fp16"
    returns:     [1, N, 199, 1]
    """
    if route == "12_bf16":
        return _run_12(in_3, in_1, in_0, in_2)
    elif route == "16_bf16":
        return _run_16(in_3, in_1, in_0, in_2)
    elif route == "12_fp16":
        return _run_12(in_3, in_1, in_0, in_2)
    else:  # "16_fp16"
        return _run_16(in_3, in_1, in_0, in_2)