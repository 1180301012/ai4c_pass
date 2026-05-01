"""
Shared Triton kernels and dispatch wrapper for SwinTransformer patch embedding.

Fuses: layer_norm + dropout(p=0, no-op) into a single kernel.
Handles non-contiguous transposed input [1, N, C] with strides [N*C, 1, N].
Writes a contiguous output [1, N, C] for faster downstream ops.
"""

import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Kernel for C = 96   (large model: N=65536)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128, 'BLOCK_P': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_P': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_P': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_P': 16}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_P': 32}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_P': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_P': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'BLOCK_P': 32}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _fused_ln_c96_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    stride_xN,   # =1 for transposed input
    stride_xC,   # =N for transposed input
    C:       tl.constexpr,
    eps:     tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    base_p = tl.program_id(0) * BLOCK_P
    c_idx  = tl.arange(0, BLOCK_C)
    p_idx  = tl.arange(0, BLOCK_P)
    c_mask = c_idx < C
    p_mask = (base_p + p_idx) < N

    # Load [BLOCK_C, BLOCK_P] — inner dim=BLOCK_P, stride=stride_xN=1 → coalesced
    load_off = c_idx[:, None] * stride_xC + (base_p + p_idx[None, :]) * stride_xN
    x = tl.load(x_ptr + load_off,
                 mask=c_mask[:, None] & p_mask[None, :],
                 other=0.0).to(tl.float32)

    # Layer norm over channels (axis=0); zero padding channels for correct variance
    mean = tl.sum(x, axis=0) / C
    diff = x - mean[None, :]
    diff = tl.where(c_mask[:, None], diff, 0.0)
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var[None, :] + eps)

    w = tl.load(w_ptr + c_idx, mask=c_mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
    y = diff * rstd * w[:, None] + b[:, None]   # [BLOCK_C, BLOCK_P]

    # Transpose → [BLOCK_P, BLOCK_C] then cast; store with inner dim=BLOCK_C → coalesced
    if IS_BF16:
        y_t = tl.trans(y).to(tl.bfloat16)
    else:
        y_t = tl.trans(y).to(tl.float16)

    store_off = (base_p + p_idx[:, None]) * C + c_idx[None, :]
    tl.store(out_ptr + store_off, y_t,
             mask=p_mask[:, None] & c_mask[None, :])


# ──────────────────────────────────────────────────────────────────────────────
# Kernel for C = 16   (small model: N=256)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16, 'BLOCK_P': 16},  num_warps=1),
        triton.Config({'BLOCK_C': 16, 'BLOCK_P': 32},  num_warps=2),
        triton.Config({'BLOCK_C': 16, 'BLOCK_P': 64},  num_warps=2),
        triton.Config({'BLOCK_C': 16, 'BLOCK_P': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_P': 256}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _fused_ln_c16_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    stride_xN,
    stride_xC,
    C:       tl.constexpr,   # 16
    eps:     tl.constexpr,
    BLOCK_C: tl.constexpr,   # 16
    BLOCK_P: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    base_p = tl.program_id(0) * BLOCK_P
    c_idx  = tl.arange(0, BLOCK_C)
    p_idx  = tl.arange(0, BLOCK_P)
    c_mask = c_idx < C
    p_mask = (base_p + p_idx) < N

    # Coalesced load [BLOCK_C, BLOCK_P] — inner dim stride stride_xN=1
    load_off = c_idx[:, None] * stride_xC + (base_p + p_idx[None, :]) * stride_xN
    x = tl.load(x_ptr + load_off,
                 mask=c_mask[:, None] & p_mask[None, :],
                 other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / C
    diff = x - mean[None, :]
    diff = tl.where(c_mask[:, None], diff, 0.0)
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var[None, :] + eps)

    w = tl.load(w_ptr + c_idx, mask=c_mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
    y = diff * rstd * w[:, None] + b[:, None]

    if IS_BF16:
        y_t = tl.trans(y).to(tl.bfloat16)
    else:
        y_t = tl.trans(y).to(tl.float16)

    # Coalesced store [BLOCK_P, BLOCK_C] — inner dim stride 1
    store_off = (base_p + p_idx[:, None]) * C + c_idx[None, :]
    tl.store(out_ptr + store_off, y_t,
             mask=p_mask[:, None] & c_mask[None, :])


# ──────────────────────────────────────────────────────────────────────────────
# Python-level wrappers
# ──────────────────────────────────────────────────────────────────────────────

def _run_c96(x, weight, bias):
    _, N, C = x.shape
    stride_xN = x.stride(1)
    stride_xC = x.stride(2)
    IS_BF16 = (x.dtype == torch.bfloat16)
    out = torch.empty((1, N, C), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_P']),)
    _fused_ln_c96_kernel[grid](
        x, weight, bias, out,
        N, stride_xN, stride_xC,
        C=96, eps=1e-05, IS_BF16=IS_BF16,
    )
    return out


def _run_c16(x, weight, bias):
    _, N, C = x.shape
    stride_xN = x.stride(1)
    stride_xC = x.stride(2)
    IS_BF16 = (x.dtype == torch.bfloat16)
    out = torch.empty((1, N, C), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_P']),)
    _fused_ln_c16_kernel[grid](
        x, weight, bias, out,
        N, stride_xN, stride_xC,
        C=16, eps=1e-05, IS_BF16=IS_BF16,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shared dispatch wrapper — SAME object returned by both pass files
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def swin_fused_dispatch(x, weight, bias, route):
    """
    Route-dispatched fused layer_norm wrapper.
    route="c96"  →  C=96 large model (returns single tensor)
    route="c16"  →  C=16 small model (returns single tensor)
    """
    if route == "c96":
        return _run_c96(x, weight, bias)
    elif route == "c16":
        return _run_c16(x, weight, bias)
    else:
        return _run_c96(x, weight, bias)