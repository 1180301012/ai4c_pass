"""
Shared fused operations for all optimization passes.
All passes use the universal_fused_op wrapper as replacement_func.

Routes:
  'dw_conv3x3_gelu_*' : fused depthwise 3x3 conv + GELU (no dropout, p=0 training=False)
  'gelu_default'       : GELU-only (no dropout)
  'gelu_approx_none'   : GELU(approximate='none')-only (no dropout)
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused depthwise 3x3 conv + GELU kernel  (flat spatial-index approach)
#
# Each program handles BLOCK_HW consecutive spatial positions for one (n,c).
# BLOCK_HW is chosen at launch to minimise masked-element waste:
#   H*W ≤  64  →  BLOCK_HW=64   (≥ 76 % utilisation)
#   H*W > 64   →  BLOCK_HW=512  (up to 87 % utilisation for 56×56)
#
# Assumes: padding=(1,1), stride=(1,1), dilation=(1,1), groups=C (depthwise)
# Input  : [N, C, H, W] contiguous NCHW
# Weight : [C, 1, 3, 3] (9 elements per channel, contiguous)
# Bias   : [C]
# Output : [N, C, H, W] contiguous NCHW
# ---------------------------------------------------------------------------
@triton.jit
def _dw_conv3x3_gelu_kernel(
    x_ptr, wt_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    BLOCK_HW: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_nc = tl.program_id(0)   # n*C + c
    pid_hw = tl.program_id(1)   # spatial-block index

    n = pid_nc // C
    c = pid_nc % C

    # Load 9 depthwise weights for channel c  (wt[c,0,kh,kw] @ offset c*9+kh*3+kw)
    wt_base = c * 9
    wt0 = tl.load(wt_ptr + wt_base + 0).to(tl.float32)
    wt1 = tl.load(wt_ptr + wt_base + 1).to(tl.float32)
    wt2 = tl.load(wt_ptr + wt_base + 2).to(tl.float32)
    wt3 = tl.load(wt_ptr + wt_base + 3).to(tl.float32)
    wt4 = tl.load(wt_ptr + wt_base + 4).to(tl.float32)
    wt5 = tl.load(wt_ptr + wt_base + 5).to(tl.float32)
    wt6 = tl.load(wt_ptr + wt_base + 6).to(tl.float32)
    wt7 = tl.load(wt_ptr + wt_base + 7).to(tl.float32)
    wt8 = tl.load(wt_ptr + wt_base + 8).to(tl.float32)

    bias_val = tl.load(bias_ptr + c).to(tl.float32)

    # Flat spatial tile
    hw_start = pid_hw * BLOCK_HW
    hw_idx   = hw_start + tl.arange(0, BLOCK_HW)
    valid    = hw_idx < H * W

    # Spatial coordinates (integer division / modulo)
    h_coord = hw_idx // W
    w_coord = hw_idx % W

    # Base offset for (n, c) feature map
    base = (n * C + c) * H * W

    # Neighbour coordinates
    h_m1 = h_coord - 1
    h_p1 = h_coord + 1
    w_m1 = w_coord - 1
    w_p1 = w_coord + 1

    # Boundary masks
    m_ht = valid & (h_m1 >= 0)
    m_hb = valid & (h_p1 < H)
    m_wl = valid & (w_m1 >= 0)
    m_wr = valid & (w_p1 < W)

    # 9 neighbourhood loads (masked, zero-padding at boundaries)
    x00 = tl.load(x_ptr + base + h_m1*W + w_m1, mask=m_ht & m_wl, other=0.0).to(tl.float32)
    x01 = tl.load(x_ptr + base + h_m1*W + w_coord, mask=m_ht,      other=0.0).to(tl.float32)
    x02 = tl.load(x_ptr + base + h_m1*W + w_p1, mask=m_ht & m_wr, other=0.0).to(tl.float32)
    x10 = tl.load(x_ptr + base + h_coord*W + w_m1, mask=valid & m_wl, other=0.0).to(tl.float32)
    x11 = tl.load(x_ptr + base + h_coord*W + w_coord, mask=valid,     other=0.0).to(tl.float32)
    x12 = tl.load(x_ptr + base + h_coord*W + w_p1, mask=valid & m_wr, other=0.0).to(tl.float32)
    x20 = tl.load(x_ptr + base + h_p1*W + w_m1, mask=m_hb & m_wl, other=0.0).to(tl.float32)
    x21 = tl.load(x_ptr + base + h_p1*W + w_coord, mask=m_hb,      other=0.0).to(tl.float32)
    x22 = tl.load(x_ptr + base + h_p1*W + w_p1, mask=m_hb & m_wr, other=0.0).to(tl.float32)

    # Depthwise 3×3 convolution
    acc = (x00*wt0 + x01*wt1 + x02*wt2 +
           x10*wt3 + x11*wt4 + x12*wt5 +
           x20*wt6 + x21*wt7 + x22*wt8 + bias_val)

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gelu_out = acc * 0.5 * (1.0 + tl.math.erf(acc * INV_SQRT2))

    # Cast to input dtype
    if IS_FP16:
        result = gelu_out.to(tl.float16)
    elif IS_BF16:
        result = gelu_out.to(tl.bfloat16)
    else:
        result = gelu_out  # float32

    tl.store(out_ptr + base + h_coord*W + w_coord, result, mask=valid)


# ---------------------------------------------------------------------------
# GELU-only kernel (fallback for fastvit / gelu-only patterns)
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_fwd_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    x_f32 = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out_f32 = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * INV_SQRT2))
    out = out_f32.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask, eviction_policy='evict_first')


# ---------------------------------------------------------------------------
# Universal replacement wrapper — shared by ALL passes via routing
# ---------------------------------------------------------------------------
@torch.fx.wrap
def universal_fused_op(x, ctx1, ctx2, route):
    """
    Universal dispatcher for all fused passes.

    'dw_conv3x3_gelu_*': ctx1=weight[C,1,3,3], ctx2=bias[C]  → fused dw-conv + GELU
    'gelu_default'/'gelu_approx_none': ctx1=ctx2=x (dummy)    → GELU-only
    """
    if route == 'gelu_default' or route == 'gelu_approx_none':
        N_elem = x.numel()
        out = torch.empty_like(x)
        BLOCK_SIZE = 4096
        _gelu_fwd_kernel[((N_elem + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
            x, out, N_elem, BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
        )
        return out

    # Fused depthwise 3x3 conv + GELU
    weight = ctx1
    bias   = ctx2
    xN = x.shape[0]
    xC = x.shape[1]
    xH = x.shape[2]
    xW = x.shape[3]
    out = torch.empty_like(x)
    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)

    HW = xH * xW
    # Choose BLOCK_HW to maximise utilisation:
    #   small HW (e.g. 7×7=49)  → 64   (76 % utilisation)
    #   large HW (e.g. 56×56=3136) → 512 (87 % avg utilisation across 7 blocks)
    if HW <= 64:
        BLOCK_HW = 64
        num_warps = 2
    elif HW <= 256:
        BLOCK_HW = 256
        num_warps = 4
    else:
        BLOCK_HW = 512
        num_warps = 4   # 128 threads → 8 blocks/SM vs 4 with 8 warps: better latency hiding

    num_blocks_hw = (HW + BLOCK_HW - 1) // BLOCK_HW
    _dw_conv3x3_gelu_kernel[(xN * xC, num_blocks_hw)](
        x, weight, bias, out,
        xN, xC, xH, xW,
        BLOCK_HW=BLOCK_HW,
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
        num_warps=num_warps,
    )
    return out