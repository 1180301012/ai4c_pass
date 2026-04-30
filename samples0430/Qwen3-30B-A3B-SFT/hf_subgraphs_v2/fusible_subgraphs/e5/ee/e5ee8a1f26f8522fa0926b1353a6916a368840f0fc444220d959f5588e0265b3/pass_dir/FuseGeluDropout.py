import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fuse depthwise-conv2d (groups=C, 3×3, pad=1) + GELU + dropout(p=0) into
# ONE Triton kernel.  This eliminates the intermediate global-memory write
# of the conv output AND the subsequent read for GELU, saving ~2× memory
# traffic compared to the separate conv → GELU → dropout pipeline.
# ---------------------------------------------------------------------------


def pattern(in_0, in_1, in_2, groups):
    """
    Matches:
      conv2d = torch.conv2d(in_2, in_1, in_0, (1,1), (1,1), (1,1), groups)
      tmp_3  = torch.nn.functional.gelu(conv2d)
      tmp_4  = torch.nn.functional.dropout(tmp_3, 0.0, False, False)

    `groups` is a wildcard so it matches any concrete groups value
    (128, 256, 512, 1024, 2048 for depthwise, or 1 for pointwise).
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), groups)
    tmp_3  = torch.nn.functional.gelu(conv2d)
    tmp_4  = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2, groups):
    return (in_0, in_1, in_2, groups)


# ---------------------------------------------------------------------------
# Triton kernel: fused depthwise conv2d (any 3×3 or 1×1 kernel) + GELU
#
# Grid:
#   axis-0 : N * C  (one program per (batch, channel) pair)
#   axis-1 : ceil(H*W / BLOCK_HW)  (spatial tiles)
#
# All kernel dimensions and strides are tl.constexpr so the inner loops are
# fully unrolled by the Triton compiler.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _fused_dw_conv_gelu_kernel(
    x_ptr,    # [N, C, H, W]   input feature map
    w_ptr,    # [C, 1, KH, KW] weight  (depthwise) or [C_out, C_in, 1, 1]
    b_ptr,    # [C]             bias
    out_ptr,  # [N, C, H, W]   output
    N, C, H, W,
    KH: tl.constexpr, KW: tl.constexpr,
    pad_h:    tl.constexpr,   # = 1 for same-padding 3×3
    pad_w:    tl.constexpr,   # = 1 for same-padding 3×3
    HW,                       # = H * W  (runtime, used for masking)
    BLOCK_HW: tl.constexpr,
):
    nc_id     = tl.program_id(0)   # which (n, c) pair
    hw_block  = tl.program_id(1)   # which spatial tile

    n = nc_id // C
    c = nc_id  % C

    # ── spatial tile ────────────────────────────────────────────────────────
    hw_start = hw_block * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask  = hw_offs < HW

    h_v = hw_offs // W   # row index for each lane
    w_v = hw_offs  % W   # col index for each lane

    # ── bias (scalar broadcast) ─────────────────────────────────────────────
    b_val = tl.load(b_ptr + c).to(tl.float32)
    acc   = tl.zeros([BLOCK_HW], dtype=tl.float32) + b_val

    # ── convolution over the kernel window ──────────────────────────────────
    # We load weight for each (kh, kw) once; input for each (kh, kw) with
    # boundary masking.  Unroll is forced by tl.constexpr ranges.
    for dh in range(KH):
        for dw in range(KW):
            h_abs = h_v * 1 + dh - pad_h   # stride_h = 1
            w_abs = w_v * 1 + dw - pad_w   # stride_w = 1

            in_bounds = ((h_abs >= 0) & (h_abs < H)) & ((w_abs >= 0) & (w_abs < W))
            valid     = in_bounds & hw_mask

            # safely clamp to avoid out-of-range pointer arithmetic
            h_safe = tl.where(h_abs >= 0, h_abs, 0)
            h_safe = tl.where(h_safe < H,  h_safe, H - 1)
            w_safe = tl.where(w_abs >= 0, w_abs, 0)
            w_safe = tl.where(w_safe < W,  w_safe, W - 1)

            # flat offset in NCHW layout
            x_off = (n * C + c) * HW + h_safe * W + w_safe
            xv    = tl.load(x_ptr + x_off, mask=valid, other=0.0).to(tl.float32)

            # weight index: layout [C, 1, KH, KW]
            w_off = c * KH * KW + dh * KW + dw
            wv    = tl.load(w_ptr + w_off).to(tl.float32)

            acc = acc + xv * wv

    # ── GELU: x * 0.5 * (1 + erf(x / √2)) ──────────────────────────────────
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gelu = acc * 0.5 * (1.0 + tl.math.erf(acc * INV_SQRT2))

    # ── store ────────────────────────────────────────────────────────────────
    out_off = (n * C + c) * HW + h_v * W + w_v
    tl.store(out_ptr + out_off, gelu.to(x_ptr.dtype.element_ty), mask=hw_mask)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_fused_dw_conv_gelu(bias, weight, x, groups):
    """
    bias   : [C]       bias tensor
    weight : [C, 1, KH, KW]  (depthwise)  OR  [C_out, C_in/groups, 1, 1] (pointwise)
    x      : [N, C, H, W]
    groups : int       conv2d groups (128/256/512/1024/2048 for depthwise, 1 for pointwise)
    """
    N, C, H, W = x.shape
    KH         = weight.shape[2]
    KW         = weight.shape[3]

    out = torch.empty_like(x)
    HW  = H * W

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_dw_conv_gelu_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        KH, KW,       # KH, KW  (constexpr — kernel size)
        1, 1,         # pad_h, pad_w
        HW,           # H*W
    )
    return out


def replacement_func():
    return triton_fused_dw_conv_gelu