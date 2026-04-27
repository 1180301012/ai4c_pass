import torch
import triton
import triton.language as tl


# Pattern: fuse sigmoid + bilinear interpolate + elementwise multiply
# conv_out: [B, C, H_in, W_in]  (e.g. [1, 128, 1, 4])
# in_2:     [B, C, H_out, W_out] (e.g. [1, 128, 64, 128])
def pattern(conv_out, in_2):
    tmp_2 = torch.sigmoid(conv_out)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['C', 'H_out', 'W_out'],
)
@triton.jit
def _fused_sigmoid_bilinear_mul_kernel(
    conv_ptr,   # [B, C, H_in, W_in]  – small, fits in L1
    in2_ptr,    # [B, C, H_out, W_out]
    out_ptr,    # [B, C, H_out, W_out]
    B, C,
    H_in,  W_in,
    H_out, W_out,
    scale_h, scale_w,   # fp32: H_in/H_out, W_in/W_out
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    total = B * C * H_out * W_out
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < total

    # Decompose flat index → (b, c, h_out, w_out)
    w_out_idx = offs % W_out
    rem        = offs // W_out
    h_out_idx  = rem  % H_out
    rem2       = rem  // H_out
    c_idx      = rem2 % C
    b_idx      = rem2 // C

    # ------- bilinear source coords (align_corners=False) --------
    x_in = (w_out_idx.to(tl.float32) + 0.5) * scale_w - 0.5
    y_in = (h_out_idx.to(tl.float32) + 0.5) * scale_h - 0.5

    x0 = tl.floor(x_in).to(tl.int32)
    y0 = tl.floor(y_in).to(tl.int32)

    x1_idx = tl.maximum(x0,     0)
    x2_idx = tl.maximum(tl.minimum(x0 + 1, W_in - 1), 0)
    y1_idx = tl.maximum(y0,     0)
    y2_idx = tl.maximum(tl.minimum(y0 + 1, H_in - 1), 0)

    wx = tl.minimum(tl.maximum(x_in - x0.to(tl.float32), 0.0), 1.0)
    wy = tl.minimum(tl.maximum(y_in - y0.to(tl.float32), 0.0), 1.0)

    # conv_out base for this (b, c)
    base = (b_idx * C + c_idx) * H_in * W_in

    v00 = tl.load(conv_ptr + base + y1_idx * W_in + x1_idx, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(conv_ptr + base + y1_idx * W_in + x2_idx, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(conv_ptr + base + y2_idx * W_in + x1_idx, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(conv_ptr + base + y2_idx * W_in + x2_idx, mask=mask, other=0.0).to(tl.float32)

    # Bilinear interpolation
    val = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) \
        +        wy  * ((1.0 - wx) * v10 + wx * v11)

    # Sigmoid
    sig = 1.0 / (1.0 + tl.exp(-val))

    # Load in_2 and multiply
    in2_val = tl.load(in2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    result   = sig * in2_val

    # Store in original dtype
    if IS_BF16:
        tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offs, result.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_sigmoid_bilinear_mul(conv_out, in_2):
    B,  C,  H_in,  W_in  = conv_out.shape
    _,  _,  H_out, W_out = in_2.shape

    scale_h = H_in / H_out
    scale_w = W_in / W_out

    out     = torch.empty_like(in_2)
    IS_BF16 = (in_2.dtype == torch.bfloat16)

    total = B * C * H_out * W_out
    grid  = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_sigmoid_bilinear_mul_kernel[grid](
        conv_out, in_2, out,
        B, C,
        H_in,  W_in,
        H_out, W_out,
        scale_h, scale_w,
        IS_BF16=IS_BF16,
    )
    return out


def replacement_func():
    return fused_sigmoid_bilinear_mul