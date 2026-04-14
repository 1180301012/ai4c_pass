"""
Shared Triton kernels for layer-norm with 2D-tiled coalesced access.
Used by FusedDwConvLN_768_bf16.py and FusedDwConvLN_1024_f32.py via import.

Key insight: input x has shape [1, N, C] with strides (C*N, 1, N).
  x[0, n, c] is at x_ptr + n*1 + c*N  →  stride-1 in the N direction.
The 2D-tiled kernel loads [BLOCK_C, BLOCK_N] tiles where the BLOCK_N
dimension has stride-1 access → fully COALESCED reads.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Bfloat16 / C=768 kernel  (2D tiled, coalesced)
# ---------------------------------------------------------------------------
@triton.jit
def _ln_bf16_768_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,              # number of tokens (256)
    C,              # number of channels (768)
    stride_tok,     # stride[1] of x = 1
    stride_feat,    # stride[2] of x = N = 256
    BLOCK_N: tl.constexpr,   # tokens per program  (32)
    BLOCK_C: tl.constexpr,   # channels per tile   (32)
):
    """
    Grid: [N // BLOCK_N] = 8 programs.
    Each program handles BLOCK_N tokens, iterating over all C channels
    in BLOCK_C-wide tiles with coalesced (stride-1) loads.
    Two-pass: accumulate stats then normalize.
    """
    token_start = tl.program_id(0) * BLOCK_N
    tok_ids = token_start + tl.arange(0, BLOCK_N)    # [BLOCK_N]
    tok_mask = tok_ids < N

    # ---- Pass 1: accumulate sum and sum-of-squares -------------------------
    acc_x  = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_x2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_ids = c_start + tl.arange(0, BLOCK_C)      # [BLOCK_C]
        c_mask = c_ids < C

        # offsets[c, n] = c_ids[c] * stride_feat + tok_ids[n] * stride_tok
        #              = c * N  +  n              (stride_tok=1)
        # Last dim (BLOCK_N, stride 1) is innermost → COALESCED
        offsets = c_ids[:, None] * stride_feat + tok_ids[None, :]  # [BLOCK_C, BLOCK_N]
        data = tl.load(x_ptr + offsets,
                       mask=c_mask[:, None] & tok_mask[None, :],
                       other=0.0).to(tl.float32)

        acc_x  = acc_x  + tl.sum(data,        axis=0)  # [BLOCK_N]
        acc_x2 = acc_x2 + tl.sum(data * data, axis=0)

    mean = acc_x / C
    var  = acc_x2 / C - mean * mean
    rstd = tl.rsqrt(var + 1e-5)   # [BLOCK_N]

    # ---- Pass 2: normalize and store ----------------------------------------
    for c_start in range(0, C, BLOCK_C):
        c_ids = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_ids < C

        offsets = c_ids[:, None] * stride_feat + tok_ids[None, :]
        data = tl.load(x_ptr + offsets,
                       mask=c_mask[:, None] & tok_mask[None, :],
                       other=0.0).to(tl.float32)

        w = tl.load(w_ptr + c_ids, mask=c_mask).to(tl.float32)   # [BLOCK_C]
        b = tl.load(b_ptr + c_ids, mask=c_mask).to(tl.float32)

        diff = data - mean[None, :]                                # [BLOCK_C, BLOCK_N]
        norm = diff * rstd[None, :] * w[:, None] + b[:, None]

        # Store: out[0, tok, chan] at out_ptr + tok*C + chan
        # out_offsets[c, n] = tok_ids[n]*C + c_ids[c]
        # Use col-major [BLOCK_C, BLOCK_N] store; inner dim stride is C (tokens vary slowly).
        # Acceptable: stores are not coalesced but loads dominate runtime.
        out_offsets = tok_ids[None, :] * C + c_ids[:, None]       # [BLOCK_C, BLOCK_N]
        tl.store(out_ptr + out_offsets, norm.to(tl.bfloat16),
                 mask=c_mask[:, None] & tok_mask[None, :])


# ---------------------------------------------------------------------------
# Float32 / C=1024 kernel (same structure)
# ---------------------------------------------------------------------------
@triton.jit
def _ln_f32_1024_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C,
    stride_tok, stride_feat,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token_start = tl.program_id(0) * BLOCK_N
    tok_ids = token_start + tl.arange(0, BLOCK_N)
    tok_mask = tok_ids < N

    acc_x  = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_x2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_ids = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_ids < C

        offsets = c_ids[:, None] * stride_feat + tok_ids[None, :]
        data = tl.load(x_ptr + offsets,
                       mask=c_mask[:, None] & tok_mask[None, :],
                       other=0.0).to(tl.float32)

        acc_x  = acc_x  + tl.sum(data,        axis=0)
        acc_x2 = acc_x2 + tl.sum(data * data, axis=0)

    mean = acc_x / C
    var  = acc_x2 / C - mean * mean
    rstd = tl.rsqrt(var + 1e-5)

    for c_start in range(0, C, BLOCK_C):
        c_ids = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_ids < C

        offsets = c_ids[:, None] * stride_feat + tok_ids[None, :]
        data = tl.load(x_ptr + offsets,
                       mask=c_mask[:, None] & tok_mask[None, :],
                       other=0.0).to(tl.float32)

        w = tl.load(w_ptr + c_ids, mask=c_mask).to(tl.float32)
        b = tl.load(b_ptr + c_ids, mask=c_mask).to(tl.float32)

        diff = data - mean[None, :]
        norm = diff * rstd[None, :] * w[:, None] + b[:, None]

        out_offsets = tok_ids[None, :] * C + c_ids[:, None]
        tl.store(out_ptr + out_offsets, norm,
                 mask=c_mask[:, None] & tok_mask[None, :])


# ---------------------------------------------------------------------------
# Wrappers called from the dispatch function
# ---------------------------------------------------------------------------
def _impl_ln_bf16_768(x, weight, bias):
    """
    x: [B, N, C] = [1, 256, 768] bfloat16, strides (C*N, 1, N).
    Returns out: [B, N, C] = [1, 256, 768] contiguous bfloat16.
    """
    B, N, C = x.shape
    s0, stride_tok, stride_feat = x.stride()   # (C*N, 1, N)
    out = torch.empty(B, N, C, dtype=x.dtype, device=x.device)

    BLOCK_N = 32
    BLOCK_C = 32
    grid = (N // BLOCK_N,)    # = 8

    _ln_bf16_768_kernel[grid](
        x_ptr=x, w_ptr=weight, b_ptr=bias, out_ptr=out,
        N=N, C=C,
        stride_tok=stride_tok, stride_feat=stride_feat,
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


def _impl_ln_f32_1024(x, weight, bias):
    """
    x: [B, N, C] = [1, 256, 1024] float32, strides (C*N, 1, N).
    Returns out: [B, N, C] = [1, 256, 1024] contiguous float32.
    """
    B, N, C = x.shape
    s0, stride_tok, stride_feat = x.stride()
    out = torch.empty(B, N, C, dtype=x.dtype, device=x.device)

    BLOCK_N = 32
    BLOCK_C = 32
    grid = (N // BLOCK_N,)

    _ln_f32_1024_kernel[grid](
        x_ptr=x, w_ptr=weight, b_ptr=bias, out_ptr=out,
        N=N, C=C,
        stride_tok=stride_tok, stride_feat=stride_feat,
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Shared dispatch wrapper (SAME object returned by both pass files).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_dwconv_ln_dispatch(x, weight, bias, route):
    """
    Shared replacement for layer_norm, dispatched by route string.
    route: "bf16_768" | "f32_1024"
    Returns: normalized tensor (same shape as x).
    """
    if route == "bf16_768":
        return _impl_ln_bf16_768(x, weight, bias)
    elif route == "f32_1024":
        return _impl_ln_f32_1024(x, weight, bias)