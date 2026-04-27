import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: weighted sum given PRE-SOFTMAXED weights
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _fused_weighted_sum_kernel(
    in0_ptr,
    weights_ptr,
    out_ptr,
    HW,
    # in_0 strides [batch, branch, channel]  (H*W contiguous)
    s0_b, s0_k, s0_c,
    # weights strides [batch, branch, channel]  (weights[b,k,0,c])
    sw_b, sw_k, sw_c,
    # out strides [batch, channel]  (H*W contiguous)
    so_b, so_c,
    BLOCK_HW: tl.constexpr,
):
    """
    out[b, c, hw] = w0[b,c] * in0[b,0,c,hw] + w1[b,c] * in0[b,1,c,hw]
    where w0, w1 come from the pre-softmaxed weights tensor.
    """
    pid_b  = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_start   = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    # load pre-softmaxed branch weights for (b, c)
    w_base = pid_b * sw_b + pid_c * sw_c
    w0 = tl.load(weights_ptr + w_base        ).to(tl.float32)
    w1 = tl.load(weights_ptr + w_base + sw_k ).to(tl.float32)

    # load feature-map tiles
    in0_base0 = pid_b * s0_b + pid_c * s0_c
    in0_base1 = in0_base0 + s0_k

    x0 = tl.load(in0_ptr + in0_base0 + hw_offsets, mask=hw_mask, other=0.0)
    x1 = tl.load(in0_ptr + in0_base1 + hw_offsets, mask=hw_mask, other=0.0)

    # weighted sum (fp32 accumulation for safety)
    out_f32 = w0 * x0.to(tl.float32) + w1 * x1.to(tl.float32)
    out = out_f32.to(x0.dtype)

    tl.store(out_ptr + pid_b * so_b + pid_c * so_c + hw_offsets, out, mask=hw_mask)


@torch.fx.wrap
def fused_weighted_sum(in_0, softmax_out):
    """
    Fast weighted sum with pre-computed softmax attention weights.

    in_0       : [B, 2, C, H, W]  feature maps for 2 branches
    softmax_out: [B, 2, 1, C]     softmax attention weights (sum-to-1 along dim=1)
    returns    : [B, C, H, W]     weighted sum across branches
    """
    B  = in_0.shape[0]
    C  = in_0.shape[2]
    H  = in_0.shape[3]
    W  = in_0.shape[4]
    HW = H * W

    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (B, C, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_weighted_sum_kernel[grid](
        in_0, softmax_out, out,
        HW,
        in_0.stride(0),      in_0.stride(1),      in_0.stride(2),
        softmax_out.stride(0), softmax_out.stride(1), softmax_out.stride(3),
        out.stride(0),       out.stride(1),
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Legacy kernel: fuses softmax computation as well (kept for reference)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _fused_softmax_weighted_sum_kernel(
    in0_ptr, in1_ptr, out_ptr,
    HW,
    s0_b, s0_k, s0_c,
    s1_b, s1_k, s1_c,
    so_b, so_c,
    BLOCK_HW: tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_start   = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    in1_base = pid_b * s1_b + pid_c * s1_c
    w0_raw = tl.load(in1_ptr + in1_base        ).to(tl.float32)
    w1_raw = tl.load(in1_ptr + in1_base + s1_k ).to(tl.float32)

    w_max = tl.maximum(w0_raw, w1_raw)
    e0    = tl.exp(w0_raw - w_max)
    e1    = tl.exp(w1_raw - w_max)
    inv_s = 1.0 / (e0 + e1)
    w0 = e0 * inv_s
    w1 = e1 * inv_s

    in0_base0 = pid_b * s0_b + pid_c * s0_c
    in0_base1 = in0_base0 + s0_k

    x0 = tl.load(in0_ptr + in0_base0 + hw_offsets, mask=hw_mask, other=0.0)
    x1 = tl.load(in0_ptr + in0_base1 + hw_offsets, mask=hw_mask, other=0.0)

    result_f32 = w0 * x0.to(tl.float32) + w1 * x1.to(tl.float32)
    result     = result_f32.to(x0.dtype)

    tl.store(out_ptr + pid_b * so_b + pid_c * so_c + hw_offsets, result, mask=hw_mask)


@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1):
    B  = in_0.shape[0]
    C  = in_0.shape[2]
    H  = in_0.shape[3]
    W  = in_0.shape[4]
    HW = H * W
    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)
    grid = lambda meta: (B, C, triton.cdiv(HW, meta['BLOCK_HW']))
    _fused_softmax_weighted_sum_kernel[grid](
        in_0, in_1, out,
        HW,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(3),
        out.stride(0),  out.stride(1),
    )
    return out