import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 64, 'BLOCK_C': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 64, 'BLOCK_C': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 32, 'BLOCK_C': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 32, 'BLOCK_C': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 128, 'BLOCK_C': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 64, 'BLOCK_C': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_P': 64, 'BLOCK_C': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_P': 32, 'BLOCK_C': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_P': 128, 'BLOCK_C': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_P': 64, 'BLOCK_C': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 64, 'BLOCK_C': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 128, 'BLOCK_C': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 128, 'BLOCK_C': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 256, 'BLOCK_C': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 256, 'BLOCK_C': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['C', 'K', 'HW_out'],
)
@triton.jit
def fused_conv2d_bilinear_kernel(
    x_ptr,    # [1, K, H_IN, W_IN]  -- input feature map
    w_ptr,    # [C, K, 1, 1]        -- conv weight
    b_ptr,    # [C]                 -- conv bias
    out_ptr,  # [1, C, H_OUT, W_OUT]-- output feature map
    C, K,
    H_IN, W_IN,
    H_OUT, W_OUT,
    scale_h, scale_w,
    IS_FP16: tl.constexpr,
    BLOCK_P: tl.constexpr,   # tile over output pixels
    BLOCK_C: tl.constexpr,   # tile over output channels
    BLOCK_K: tl.constexpr,   # tile over reduction (input channels)
):
    # ---- program IDs ----
    pid_p = tl.program_id(0)   # which P-tile
    pid_c = tl.program_id(1)   # which C-tile

    # ---- output pixel indices for this tile ----
    p_start  = pid_p * BLOCK_P
    c_start  = pid_c * BLOCK_C

    p_idx  = p_start + tl.arange(0, BLOCK_P)   # [BLOCK_P]
    c_idx  = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]

    # 2-D output position vectors
    h_out_v = p_idx // W_OUT   # [BLOCK_P]
    w_out_v = p_idx  % W_OUT   # [BLOCK_P]

    # ---- bilinear source coordinates (align_corners=False) ----
    h_src_v = (h_out_v.to(tl.float32) + 0.5) * scale_h - 0.5
    w_src_v = (w_out_v.to(tl.float32) + 0.5) * scale_w - 0.5

    # clamp to valid input range
    h_src0_v = tl.maximum(tl.minimum(h_src_v.to(tl.int32),       H_IN - 1), 0)
    w_src0_v = tl.maximum(tl.minimum(w_src_v.to(tl.int32),       W_IN - 1), 0)
    h1_v     = tl.maximum(tl.minimum((h_src_v + 1.0).to(tl.int32), H_IN - 1), 0)
    w1_v     = tl.maximum(tl.minimum((w_src_v + 1.0).to(tl.int32), W_IN - 1), 0)

    # fractional parts
    fh = h_src_v - h_src_v.to(tl.int32).to(tl.float32)
    fw = w_src_v - w_src_v.to(tl.int32).to(tl.float32)

    # tl.where masks for bilinear 4-tap
    lt00 = (h_src_v >= 0.0) & (w_src_v >= 0.0)
    lt01 = (h_src_v >= 0.0) & (w_src_v <  0.0)
    lt10 = (h_src_v <  0.0) & (w_src_v >= 0.0)
    lt11 = (h_src_v <  0.0) & (w_src_v <  0.0)

    # ---- accumulator ----
    acc = tl.zeros([BLOCK_P, BLOCK_C], dtype=tl.float32)

    HW_IN = H_IN * W_IN   # stride for k-dim of X

    # ---- tiled GEMM over K reduction ----
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load X at 4 bilinear source positions: [BLOCK_K, BLOCK_P]
        base_x = k_idx[:, None] * HW_IN  # [BLOCK_K, 1]
        a00_idx = base_x + h_src0_v[None, :] * W_IN + w_src0_v[None, :]   # [BLOCK_K, BLOCK_P]
        a01_idx = base_x + h_src0_v[None, :] * W_IN + w1_v[None, :]
        a10_idx = base_x + h1_v[None, :]     * W_IN + w_src0_v[None, :]
        a11_idx = base_x + h1_v[None, :]     * W_IN + w1_v[None, :]

        a00 = tl.load(x_ptr + a00_idx,
                      mask=(k_idx[:, None] < K) & (p_idx[None, :] < HW_out), other=0.0)
        a01 = tl.load(x_ptr + a01_idx,
                      mask=(k_idx[:, None] < K) & (p_idx[None, :] < HW_out), other=0.0)
        a10 = tl.load(x_ptr + a10_idx,
                      mask=(k_idx[:, None] < K) & (p_idx[None, :] < HW_out), other=0.0)
        a11 = tl.load(x_ptr + a11_idx,
                      mask=(k_idx[:, None] < K) & (p_idx[None, :] < HW_out), other=0.0)

        # Load W[c, k]  ->  [BLOCK_C, BLOCK_K]
        w_idx_2d = c_idx[:, None] * K + k_idx[None, :]
        W_blk = tl.load(w_ptr + w_idx_2d,
                        mask=(c_idx[:, None] < C) & (k_idx[None, :] < K), other=0.0)

        # acc[BLOCK_P, BLOCK_C] += X_blk[BLOCK_P, BLOCK_K] @ W_blk.T[BLOCK_K, BLOCK_C]
        # X_blk from a00/a01/a10/a11 is [BLOCK_K, BLOCK_P]
        acc += tl.dot(a00, tl.trans(W_blk), allow_tf32=False).to(tl.float32) * (1.0 - fh) * (1.0 - fw) \
             + tl.dot(a01, tl.trans(W_blk), allow_tf32=False).to(tl.float32) * (1.0 - fh) * fw \
             + tl.dot(a10, tl.trans(W_blk), allow_tf32=False).to(tl.float32) * fh * (1.0 - fw) \
             + tl.dot(a11, tl.trans(W_blk), allow_tf32=False).to(tl.float32) * fh * fw

    # ---- add bias ----
    b   = tl.load(b_ptr + c_idx, mask=c_idx < C, other=0.0)
    acc = acc + b[None, :].to(tl.float32)

    # ---- store output ----
    out_idx = c_idx[None, :] * (H_OUT * W_OUT) + h_out_v[:, None] * W_OUT + w_out_v[:, None]
    mask_st = (c_idx[None, :] < C) & (p_idx[:, None] < HW_out)

    if IS_FP16:
        tl.store(out_ptr + out_idx, acc.to(tl.float16), mask=mask_st)
    else:
        tl.store(out_ptr + out_idx, acc.to(tl.bfloat16), mask=mask_st)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv2d_bilinear(x, weight, bias, output_size):
    """
    Fused 1x1-Conv2d + Bilinear-Interp-to-512x512.
    x      : [N, K, H_IN, W_IN]
    weight : [C, K, 1, 1]
    bias   : [C]
    """
    N, K, H_IN, W_IN = x.shape
    C = weight.shape[0]
    H_OUT, W_OUT = output_size
    HW_out = H_OUT * W_OUT
    scale_h = H_IN / H_OUT
    scale_w = W_IN / W_OUT

    out     = torch.empty((N, C, H_OUT, W_OUT), dtype=x.dtype, device=x.device)
    is_fp16 = (x.dtype == torch.float16)

    grid = lambda meta: (
        (HW_out + meta['BLOCK_P'] - 1) // meta['BLOCK_P'],
        (C     + meta['BLOCK_C'] - 1) // meta['BLOCK_C'],
    )

    fused_conv2d_bilinear_kernel[grid](
        x, weight, bias, out,
        C, K, H_IN, W_IN, H_OUT, W_OUT,
        scale_h, scale_w,
        is_fp16,
    )

    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(x, weight, bias):
    """
    Match:
        conv2d = torch.conv2d(x, weight, bias, (1,1), (0,0), (1,1), 1)
        out    = F.interpolate(conv2d, (512,512), 'bilinear', align_corners=False)
    """
    conv_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    out = torch.nn.functional.interpolate(
        conv_out, size=(512, 512), mode='bilinear', align_corners=False
    )
    return out


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return fused_conv2d_bilinear