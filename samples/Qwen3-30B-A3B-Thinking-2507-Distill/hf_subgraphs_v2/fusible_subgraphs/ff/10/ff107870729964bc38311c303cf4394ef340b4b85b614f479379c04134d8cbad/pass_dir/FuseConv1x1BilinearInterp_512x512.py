import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: match 1×1 Conv2d with no groups (groups=1, padding=0 etc.)
# Matches both conv2d calls in the model:
#   path-1: conv2d([1,512,128,128], [150,512,1,1], [150]) → [1,150,128,128]
#   path-2: conv2d([1,384,32,32], [256,384,3,3],None) → dead-code [1,256,32,32]
# ──────────────────────────────────────────────────────────────────────────────
def pattern(x, weight, bias):
    out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return out


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# ──────────────────────────────────────────────────────────────────────────────
# Triton 1×1 Conv2d kernel
#   x   : [N, IC, IH, IW]   (NCHW)
#   w   : [OC, IC, 1, 1]
#   b   : [OC]
#   out : [N, OC, IH, IW]
#
# For each spatial position p = (h, w) and output channel oc:
#   out[oc, p] = Σ_ic w[oc, ic] * x[ic, p]  +  b[oc]
# This is a matmul: [IC, NUM_POS] @ [OC, IC]^T  + bias
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 512, 'BLOCK_C': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 256, 'BLOCK_C': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 256, 'BLOCK_C': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 512, 'BLOCK_C': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 1024,'BLOCK_C': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 256, 'BLOCK_C': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 512, 'BLOCK_C': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 2048,'BLOCK_C': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 1024,'BLOCK_C': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 1024,'BLOCK_C': 64},  num_stages=4, num_warps=8),
    ],
    key=['IC', 'OC', 'IH', 'IW'],
)
@triton.jit
def _conv1x1_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC, IH, IW,
    BLOCK_OC: tl.constexpr,
    BLOCK_P:  tl.constexpr,
    BLOCK_C:  tl.constexpr,
):
    pid_oc = tl.program_id(0)
    pid_p  = tl.program_id(1)

    oc_range = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    p_range  = pid_p  * BLOCK_P  + tl.arange(0, BLOCK_P)

    # Accumulator [BLOCK_OC, BLOCK_P] in float32
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    for ic_s in range(0, IC, BLOCK_C):
        ic_range = ic_s + tl.arange(0, BLOCK_C)

        # Load x slice: [BLOCK_C, BLOCK_P]
        # x[n=0, ic, ih, iw] at ic * IH * IW + ih * IW + iw
        base_x = ic_range[:, None] * (IH * IW) + p_range[None, :]
        x_slice = tl.load(
            x_ptr + base_x,
            mask=(ic_range[:, None] < IC) & (p_range[None, :] < IH * IW),
            other=0.0,
        ).to(tl.float32)

        # Load w slice: [BLOCK_OC, BLOCK_C]
        # w[oc, ic, 0, 0] at oc * IC + ic
        w_slice = tl.load(
            w_ptr + oc_range[:, None] * IC + ic_range[None, :],
            mask=(oc_range[:, None] < OC) & (ic_range[None, :] < IC),
            other=0.0,
        ).to(tl.float32)

        # acc += w @ x: [BLOCK_OC, BLOCK_C] @ [BLOCK_C, BLOCK_P]
        acc = tl.dot(w_slice, x_slice, acc, allow_tf32=False)

    # Add bias
    b = tl.load(b_ptr + oc_range, mask=oc_range < OC, other=0.0).to(tl.float32)
    acc = acc + b[:, None]

    # Store: out[0, oc, ih, iw] at oc * IH * IW + ih * IW + iw
    out_offs = oc_range[:, None] * (IH * IW) + p_range[None, :]
    tl.store(
        out_ptr + out_offs,
        acc,
        mask=(oc_range[:, None] < OC) & (p_range[None, :] < IH * IW),
    )


@torch.fx.wrap
def triton_conv1x1(x, weight, bias):
    N, IC, IH, IW = x.shape
    OC = weight.shape[0]
    out = torch.empty((N, OC, IH, IW), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(OC, META['BLOCK_OC']),
        triton.cdiv(IH * IW, META['BLOCK_P']),
    )

    _conv1x1_kernel[grid](
        x, weight, bias, out,
        N, IC, OC, IH, IW,
    )
    return out


def replacement_func():
    return triton_conv1x1

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16,  'BLOCK_P': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_C': 32,  'BLOCK_P': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_C': 64,  'BLOCK_P': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 16,  'BLOCK_P': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 32,  'BLOCK_P': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C': 64,  'BLOCK_P': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_C': 16,  'BLOCK_P': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 32,  'BLOCK_P': 256}, num_stages=3, num_warps=8),
    ],
    key=['C', 'IH', 'IW'],
)
@triton.jit
def _bilinear_upsample_kernel(
    x_ptr,    # [N, C, IH, IW]
    out_ptr,  # [N, C, OH, OW]
    C, IH, IW,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_p = tl.program_id(1)

    c_start = pid_c * BLOCK_C
    p_start = pid_p * BLOCK_P

    c_range = c_start + tl.arange(0, BLOCK_C)
    p_range = p_start + tl.arange(0, BLOCK_P)

    # scale factors
    scale_h = IH / OH
    scale_w = IW / OW

    oh = p_range // OW
    ow = p_range  % OW

    ih_l = tl.floor(oh * scale_h - 0.5).to(tl.int32)
    iw_l = tl.floor(ow * scale_w - 0.5).to(tl.int32)

    ih_h = ih_l + 1
    iw_h = iw_l + 1

    frac_h = oh * scale_h - 0.5 - ih_l.to(tl.float32)
    frac_w = ow * scale_w - 0.5 - iw_l.to(tl.float32)

    ih_l = tl.maximum(0, tl.minimum(ih_l, IH - 1))
    ih_h = tl.maximum(0, tl.minimum(ih_h, IH - 1))
    iw_l = tl.maximum(0, tl.minimum(iw_l, IW - 1))
    iw_h = tl.maximum(0, tl.minimum(iw_h, IW - 1))

    # x[0, c, ih, iw] = c * IH * IW + ih * IW + iw
    base = c_range * (IH * IW)

    v00 = tl.load(x_ptr + base[:, None] + ih_l[None, :] * IW + iw_l[None, :],
                  mask=(c_range[:, None] < C) & (p_range[None, :] < OH * OW), other=0.0).to(tl.float32)
    v01 = tl.load(x_ptr + base[:, None] + ih_l[None, :] * IW + iw_h[None, :],
                  mask=(c_range[:, None] < C) & (p_range[None, :] < OH * OW), other=0.0).to(tl.float32)
    v10 = tl.load(x_ptr + base[:, None] + ih_h[None, :] * IW + iw_l[None, :],
                  mask=(c_range[:, None] < C) & (p_range[None, :] < OH * OW), other=0.0).to(tl.float32)
    v11 = tl.load(x_ptr + base[:, None] + ih_h[None, :] * IW + iw_h[None, :],
                  mask=(c_range[:, None] < C) & (p_range[None, :] < OH * OW), other=0.0).to(tl.float32)

    result = ((1.0 - frac_h[None, :]) * (1.0 - frac_w[None, :]) * v00 +
              (1.0 - frac_h[None, :]) *        frac_w[None, :]  * v01 +
                     frac_h[None, :]  * (1.0 - frac_w[None, :]) * v10 +
                     frac_h[None, :]) *        frac_w[None, :]  * v11

    out_base = c_range * (OH * OW)
    tl.store(out_ptr + out_base[:, None] + p_range[None, :],
             result,
             mask=(c_range[:, None] < C) & (p_range[None, :] < OH * OW))


@torch.fx.wrap
def triton_bilinear_upsample(x):
    N, C, IH, IW = x.shape
    OH, OW = 512, 512

    out = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(C, META['BLOCK_C']),
        triton.cdiv(OH * OW, META['BLOCK_P']),
    )

    _bilinear_upsample_kernel[grid](
        x, out,
        C, IH, IW,
        OH=OH, OW=OW,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Triton fused kernel: 1×1 Conv2d + bilinear upsample to (512, 512)
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 128,  'BLOCK_C': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 64,   'BLOCK_C': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 256,  'BLOCK_C': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 128,  'BLOCK_C': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 64,   'BLOCK_C': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 512,  'BLOCK_C': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_P': 256,  'BLOCK_C': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 128,  'BLOCK_C': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_P': 1024, 'BLOCK_C': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_P': 256,  'BLOCK_C': 64},  num_stages=3, num_warps=8),
    ],
    key=['IC', 'OC', 'IH', 'IW'],
)
@triton.jit
def _fused_conv1x1_bilinear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC, IH, IW,
    OH: tl.constexpr, OW: tl.constexpr, IC_: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_oc = tl.program_id(0)
    pid_p  = tl.program_id(1)
    oc_range = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    p_range  = pid_p  * BLOCK_P  + tl.arange(0, BLOCK_P)

    scale_h = IH / OH
    scale_w = IW / OW
    oh = p_range // OW
    ow = p_range  % OW
    ih_l = tl.floor(oh * scale_h - 0.5).to(tl.int32)
    iw_l = tl.floor(ow * scale_w - 0.5).to(tl.int32)
    ih_h = ih_l + 1
    iw_h = iw_l + 1
    frac_h = oh * scale_h - 0.5 - ih_l.to(tl.float32)
    frac_w = ow * scale_w - 0.5 - iw_l.to(tl.float32)
    ih_l = tl.maximum(0, tl.minimum(ih_l, IH - 1))
    ih_h = tl.maximum(0, tl.minimum(ih_h, IH - 1))
    iw_l = tl.maximum(0, tl.minimum(iw_l, IW - 1))
    iw_h = tl.maximum(0, tl.minimum(iw_h, IW - 1))

    A = tl.zeros((IC_, BLOCK_P), dtype=tl.float32)
    for ic_s in range(0, IC, BLOCK_C):
        ic_range = ic_s + tl.arange(0, BLOCK_C)
        base = ic_range * (IH * IW)
        idx00 = base + ih_l * IW + iw_l
        idx01 = base + ih_l * IW + iw_h
        idx10 = base + ih_h * IW + iw_l
        idx11 = base + ih_h * IW + iw_h
        x00 = tl.load(x_ptr + idx00[None, :], mask=ic_range[None, :] < IC, other=0.0).to(tl.float32)
        x01 = tl.load(x_ptr + idx01[None, :], mask=ic_range[None, :] < IC, other=0.0).to(tl.float32)
        x10 = tl.load(x_ptr + idx10[None, :], mask=ic_range[None, :] < IC, other=0.0).to(tl.float32)
        x11 = tl.load(x_ptr + idx11[None, :], mask=ic_range[None, :] < IC, other=0.0).to(tl.float32)
        A += (1.0 - frac_h[None, :]) * (1.0 - frac_w[None, :]) * x00
        A += (1.0 - frac_h[None, :]) *        frac_w[None, :]  * x01
        A +=        frac_h[None, :]  * (1.0 - frac_w[None, :]) * x10
        A +=        frac_h[None, :]  *        frac_w[None, :]  * x11

    w = tl.load(w_ptr + oc_range[:, None] * IC_ + tl.arange(0, IC_),
                mask=(oc_range[:, None] < OC) & (tl.arange(0, IC_)[None, :] < IC), other=0.0).to(tl.float32)
    acc = tl.dot(w, A, allow_tf32=False)
    b = tl.load(b_ptr + oc_range, mask=oc_range < OC, other=0.0).to(tl.float32)
    acc = acc + b[:, None]
    out_offs = oc_range[:, None] * (OH * OW) + p_range[None, :]
    tl.store(out_ptr + out_offs, acc,
             mask=(oc_range[:, None] < OC) & (p_range[None, :] < OH * OW))


@torch.fx.wrap
def fused_conv1x1_bilinear(x, weight, bias):
    N, IC, IH, IW = x.shape
    OC = weight.shape[0]
    OH, OW = 512, 512
    out = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(OC, META['BLOCK_OC']),
                         triton.cdiv(OH * OW, META['BLOCK_P']))
    _fused_conv1x1_bilinear_kernel[grid](
        x, weight, bias, out, N, IC, OC, IH, IW, OH=OH, OW=OW, IC_=IC)
    return out


def replacement_func():
    return triton_conv1x1