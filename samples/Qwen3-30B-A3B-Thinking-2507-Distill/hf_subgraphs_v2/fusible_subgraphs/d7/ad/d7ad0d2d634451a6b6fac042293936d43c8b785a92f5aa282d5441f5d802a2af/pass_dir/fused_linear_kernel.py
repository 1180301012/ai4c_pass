"""
Shared Triton kernels + dispatch wrapper for:
1. linear(x, w, b) → permute(0,2,1) → reshape(B,-1,16,16)
2. bilinear upsample from [B,C,16,16] → [B,C,128,128]

All pass files return `dispatch_fused_linear` from replacement_func().
The route string selects which kernel to run.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused GEMM + direct write in [B, C, S] layout
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # Large tiles – match cuBLAS tile sizes (best for B=64,12,24)
        triton.Config({'BLOCK_S': 64, 'BLOCK_K': 64, 'BLOCK_C': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_S': 64, 'BLOCK_K': 64, 'BLOCK_C': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 64, 'BLOCK_K': 64, 'BLOCK_C': 128}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_S': 64, 'BLOCK_K': 32, 'BLOCK_C': 128}, num_stages=4, num_warps=4),
        # Medium tiles
        triton.Config({'BLOCK_S': 32, 'BLOCK_K': 64, 'BLOCK_C': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 32, 'BLOCK_K': 64, 'BLOCK_C': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 64, 'BLOCK_K': 64, 'BLOCK_C': 32},  num_stages=4, num_warps=4),
        # Small tiles – for B=2,8
        triton.Config({'BLOCK_S': 32, 'BLOCK_K': 64, 'BLOCK_C': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_S': 32, 'BLOCK_K': 32, 'BLOCK_C': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 16, 'BLOCK_K': 64, 'BLOCK_C': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 16, 'BLOCK_K': 64, 'BLOCK_C': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_S': 16, 'BLOCK_K': 32, 'BLOCK_C': 64},  num_stages=4, num_warps=4),
    ],
    key=['B', 'S', 'K', 'C'],
)
@triton.jit
def _fused_linear_permute_reshape_kernel(
    x_ptr,      # [B, S, K] – input (contiguous)
    w_ptr,      # [C, K]    – weight (contiguous)
    bias_ptr,   # [C]       – bias (contiguous)
    out_ptr,    # [B, C, S] – output, strides (C*S, S, 1)
    B, S, K, C,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Grid: (B, ceil(C/BLOCK_C), ceil(S/BLOCK_S))
    out[b, c, s] = sum_k x[b,s,k] * w[c,k] + bias[c]
    Weight loaded as (BLOCK_K, BLOCK_C) – coalesced along k for fixed c.
    """
    b      = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_s  = tl.program_id(2)

    s_base = pid_s * BLOCK_S
    c_base = pid_c * BLOCK_C

    s_offs = s_base + tl.arange(0, BLOCK_S)
    c_offs = c_base + tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)

        # x tile (BLOCK_S, BLOCK_K) – coalesced along k; evict_first (changes per b)
        x_offs = b * S * K + s_offs[:, None] * K + k_offs[None, :]
        x_mask = (s_offs[:, None] < S) & (k_offs[None, :] < K)
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0, eviction_policy='evict_first')

        # w tile (BLOCK_K, BLOCK_C) – coalesced along k for fixed c; evict_last (shared across b)
        w_offs = c_offs[None, :] * K + k_offs[:, None]
        w_mask = (c_offs[None, :] < C) & (k_offs[:, None] < K)
        w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0, eviction_policy='evict_last')

        acc += tl.dot(x, w, allow_tf32=True)

    bias = tl.load(bias_ptr + c_offs, mask=c_offs < C, other=0.0)
    acc += bias[None, :]

    out_offs = b * C * S + c_offs[None, :] * S + s_offs[:, None]
    out_mask = (s_offs[:, None] < S) & (c_offs[None, :] < C)
    tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Kernel 2: Bilinear upsample [B,C,H_IN,W_IN] → [B,C,H_OUT,W_OUT]
# Specialized for align_corners=False, H_IN/W_IN = 16, H_OUT/W_OUT = 128
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_COL': 128, 'BLOCK_ROW_OUT': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_COL': 64,  'BLOCK_ROW_OUT': 2}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_COL': 32,  'BLOCK_ROW_OUT': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_COL': 16,  'BLOCK_ROW_OUT': 8}, num_stages=2, num_warps=4),
    ],
    key=['B', 'C', 'H_IN', 'W_IN', 'H_OUT', 'W_OUT'],
)
@triton.jit
def _bilinear_upsample_kernel(
    src_ptr,    # [B, C, H_IN, W_IN] contiguous
    dst_ptr,    # [B, C, H_OUT, W_OUT] contiguous
    B, C, H_IN, W_IN, H_OUT, W_OUT,
    BLOCK_COL: tl.constexpr,
    BLOCK_ROW_OUT: tl.constexpr,
):
    """
    Grid: (B, C, H_OUT // BLOCK_ROW_OUT, W_OUT // BLOCK_COL)
    Each program writes one [BLOCK_ROW_OUT, BLOCK_COL] output tile.
    Source coordinates use align_corners=False.
    """
    b        = tl.program_id(0)
    c        = tl.program_id(1)
    pid_row  = tl.program_id(2)
    pid_col  = tl.program_id(3)

    h_out_base = pid_row * BLOCK_ROW_OUT
    w_out_base = pid_col * BLOCK_COL

    h_out_offs = h_out_base + tl.arange(0, BLOCK_ROW_OUT)  # (BLOCK_ROW_OUT,)
    w_out_offs = w_out_base + tl.arange(0, BLOCK_COL)      # (BLOCK_COL,)

    # Scale factors
    scale_h = H_IN.to(tl.float32) / H_OUT.to(tl.float32)
    scale_w = W_IN.to(tl.float32) / W_OUT.to(tl.float32)

    # Source coordinates (align_corners=False)
    src_h = (h_out_offs[:, None] + 0.5) * scale_h - 0.5  # (BLOCK_ROW_OUT, 1)
    src_w = (w_out_offs[None, :] + 0.5) * scale_w - 0.5  # (1, BLOCK_COL)

    # Floor (handles negative values correctly: floor(-0.4375) = -1)
    h0_f = tl.floor(src_h)
    w0_f = tl.floor(src_w)

    # Fractional parts (0.0 ≤ fh, fw < 1.0)
    fh = src_h - h0_f
    fw = src_w - w0_f

    h0 = h0_f.to(tl.int32)
    w0 = w0_f.to(tl.int32)

    h1 = h0 + 1
    w1 = w0 + 1

    # Clamp to valid range [0, H_IN-1] / [0, W_IN-1]
    h0_c = tl.maximum(0, tl.minimum(H_IN - 1, h0))
    h1_c = tl.maximum(0, tl.minimum(H_IN - 1, h1))
    w0_c = tl.maximum(0, tl.minimum(W_IN - 1, w0))
    w1_c = tl.maximum(0, tl.minimum(W_IN - 1, w1))

    # Flat source offsets: src[b, c, h, w] = b*C*H_IN*W_IN + c*H_IN*W_IN + h*W_IN + w
    base = b * C * H_IN * W_IN + c * H_IN * W_IN

    v00 = tl.load(src_ptr + base + h0_c * W_IN + w0_c).to(tl.float32)  # (BLOCK_ROW_OUT, BLOCK_COL)
    v01 = tl.load(src_ptr + base + h0_c * W_IN + w1_c).to(tl.float32)
    v10 = tl.load(src_ptr + base + h1_c * W_IN + w0_c).to(tl.float32)
    v11 = tl.load(src_ptr + base + h1_c * W_IN + w1_c).to(tl.float32)

    # Bilinear interpolation
    result = (1.0 - fh) * (1.0 - fw) * v00 \
           + (1.0 - fh) * fw          * v01 \
           + fh          * (1.0 - fw) * v10 \
           + fh          * fw          * v11

    # Write to dst[b, c, h_out, w_out] = b*C*H_OUT*W_OUT + c*H_OUT*W_OUT + h_out*W_OUT + w_out
    dst_base = b * C * H_OUT * W_OUT + c * H_OUT * W_OUT
    dst_offs = dst_base + h_out_offs[:, None] * W_OUT + w_out_offs[None, :]
    tl.store(dst_ptr + dst_offs, result.to(dst_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Shared dispatch wrapper – ALL pass files return THIS from replacement_func()
# Signature: (in_0, in_1, in_2, route)
#   in_0 = bias (or empty for bilinear)
#   in_1 = weight (or empty for bilinear)
#   in_2 = input tensor (x for linear; src for bilinear)
#   route = dispatch key
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_fused_linear(in_0, in_1, in_2, route):
    if route == "bilinear":
        # in_2 = [B, C, H_IN, W_IN] contiguous
        B    = in_2.shape[0]
        C    = in_2.shape[1]
        H_IN = in_2.shape[2]
        W_IN = in_2.shape[3]
        H_OUT, W_OUT = 128, 128
        out = torch.empty((B, C, H_OUT, W_OUT), dtype=in_2.dtype, device=in_2.device)
        grid = lambda meta: (
            B,
            C,
            triton.cdiv(H_OUT, meta['BLOCK_ROW_OUT']),
            triton.cdiv(W_OUT, meta['BLOCK_COL']),
        )
        _bilinear_upsample_kernel[grid](
            in_2, out,
            B, C, H_IN, W_IN, H_OUT, W_OUT,
        )
        return out
    else:
        # in_0 = bias [C], in_1 = weight [C, K], in_2 = x [B, S, K]
        B  = in_2.shape[0]
        S  = in_2.shape[1]
        K  = in_2.shape[2]
        C  = in_1.shape[0]
        H, W = 16, 16
        out = torch.empty((B, C, H, W), dtype=in_2.dtype, device=in_2.device)
        grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']), triton.cdiv(S, meta['BLOCK_S']))
        _fused_linear_permute_reshape_kernel[grid](
            in_2, in_1, in_0, out,
            B, S, K, C,
        )
        return out