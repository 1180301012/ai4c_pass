import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the post-conv2d subgraph
#   interp(in4, 64x64) -> sigmoid -> * in3
#   sigmoid(conv) -> * in2 -> interp(64x64)
#   add the two paths
# ---------------------------------------------------------------------------

def pattern(conv2d_out, in_2, in_3, in_4):
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.sigmoid(conv2d_out)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(conv2d_out, in_2, in_3, in_4):
    return (conv2d_out, in_2, in_3, in_4)


# ---------------------------------------------------------------------------
# Triton kernel:  fuses bilinear upsample x2, sigmoid x2, mul x2, add x1
#
# For each output position (b, c, h_out, w_out) in [B, C, 64, 64]:
#
#  Path A:  interp(in4)[b,c,h_out,w_out]  ->  sigmoid  ->  * in3[b,c,h_out,w_out]
#  Path B:  sigmoid(conv[b,c,h_in,w_in])  *  in2[b,c,h_in,w_in]  ->  interp
#  out = A + B
#
# Bilinear coords (align_corners=False):
#   x_in = (h_out + 0.5) * (H_in / H_out) - 0.5
#
# All compute in fp32; store back in native dtype.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=2),
    ],
    key=['B'],
)
@triton.jit
def _bga_fused_kernel(
    in4_ptr,    # [B, C, H_in, W_in]  – to be upsampled for path A
    in3_ptr,    # [B, C, H_out, W_out] – multiplier for path A
    conv_ptr,   # [B, C, H_in, W_in]  – sigmoid input for path B
    in2_ptr,    # [B, C, H_in, W_in]  – multiplier for path B
    out_ptr,    # [B, C, H_out, W_out] – output
    B,
    N,          # = B * C * H_out * W_out
    C:    tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Decompose linear index → (b, c, h_out, w_out)  [NCHW order]
    w_out = (offs % W_out).to(tl.int32)
    _tmp  = offs // W_out
    h_out = (_tmp % H_out).to(tl.int32)
    _tmp  = _tmp // H_out
    c_idx = (_tmp % C).to(tl.int32)
    b_idx = (_tmp // C).to(tl.int32)

    # ---- bilinear coords (align_corners=False) ----
    # scale = H_in / H_out  (16/64 = 0.25, resolved at JIT compile time)
    x_in = (tl.cast(h_out, tl.float32) + 0.5) * (H_in / H_out) - 0.5
    y_in = (tl.cast(w_out, tl.float32) + 0.5) * (W_in / W_out) - 0.5

    x0_f = tl.floor(x_in)
    y0_f = tl.floor(y_in)
    x0 = x0_f.to(tl.int32)
    y0 = y0_f.to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # fractional weights
    wx1 = x_in - x0_f
    wx0 = 1.0 - wx1
    wy1 = y_in - y0_f
    wy0 = 1.0 - wy1

    # clamp input spatial indices
    x0c = tl.maximum(0, tl.minimum(H_in - 1, x0))
    x1c = tl.maximum(0, tl.minimum(H_in - 1, x1))
    y0c = tl.maximum(0, tl.minimum(W_in - 1, y0))
    y1c = tl.maximum(0, tl.minimum(W_in - 1, y1))

    # flat base index for small tensors [B,C,H_in,W_in]
    in_base  = (b_idx * C + c_idx) * (H_in * W_in)
    out_base = (b_idx * C + c_idx) * (H_out * W_out)

    idx_00 = in_base + x0c * W_in + y0c
    idx_01 = in_base + x0c * W_in + y1c
    idx_10 = in_base + x1c * W_in + y0c
    idx_11 = in_base + x1c * W_in + y1c

    # ---- Path A: bilinear_interp(in4) → sigmoid → * in3 ----
    a00 = tl.load(in4_ptr + idx_00, mask=mask, other=0.0).to(tl.float32)
    a01 = tl.load(in4_ptr + idx_01, mask=mask, other=0.0).to(tl.float32)
    a10 = tl.load(in4_ptr + idx_10, mask=mask, other=0.0).to(tl.float32)
    a11 = tl.load(in4_ptr + idx_11, mask=mask, other=0.0).to(tl.float32)

    interp_a = wx0*wy0*a00 + wx0*wy1*a01 + wx1*wy0*a10 + wx1*wy1*a11
    sig_a    = tl.sigmoid(interp_a)

    out_idx  = out_base + h_out * W_out + w_out
    in3_val  = tl.load(in3_ptr + out_idx, mask=mask, other=0.0).to(tl.float32)
    path_a   = in3_val * sig_a

    # ---- Path B: sigmoid(conv) * in2 → bilinear_interp ----
    c00 = tl.load(conv_ptr + idx_00, mask=mask, other=0.0).to(tl.float32)
    c01 = tl.load(conv_ptr + idx_01, mask=mask, other=0.0).to(tl.float32)
    c10 = tl.load(conv_ptr + idx_10, mask=mask, other=0.0).to(tl.float32)
    c11 = tl.load(conv_ptr + idx_11, mask=mask, other=0.0).to(tl.float32)

    i00 = tl.load(in2_ptr + idx_00, mask=mask, other=0.0).to(tl.float32)
    i01 = tl.load(in2_ptr + idx_01, mask=mask, other=0.0).to(tl.float32)
    i10 = tl.load(in2_ptr + idx_10, mask=mask, other=0.0).to(tl.float32)
    i11 = tl.load(in2_ptr + idx_11, mask=mask, other=0.0).to(tl.float32)

    t00 = tl.sigmoid(c00) * i00
    t01 = tl.sigmoid(c01) * i01
    t10 = tl.sigmoid(c10) * i10
    t11 = tl.sigmoid(c11) * i11

    interp_b = wx0*wy0*t00 + wx0*wy1*t01 + wx1*wy0*t10 + wx1*wy1*t11

    # ---- final add + dtype-aware store ----
    result = (path_a + interp_b).to(DTYPE)
    tl.store(out_ptr + out_idx, result, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_bga(conv2d_out, in_2, in_3, in_4):
    B, C, H_in, W_in = conv2d_out.shape
    H_out, W_out = 64, 64
    N = B * C * H_out * W_out

    out = torch.empty(B, C, H_out, W_out,
                      dtype=conv2d_out.dtype,
                      device=conv2d_out.device)

    dt = conv2d_out.dtype
    if dt == torch.float16:
        triton_dtype = tl.float16
    elif dt == torch.bfloat16:
        triton_dtype = tl.bfloat16
    else:
        triton_dtype = tl.float32

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    _bga_fused_kernel[grid](
        in_4, in_3, conv2d_out, in_2, out,
        B, N,
        C=C,
        H_in=H_in,   W_in=W_in,
        H_out=H_out, W_out=W_out,
        DTYPE=triton_dtype,
    )

    return out


def replacement_func():
    return fused_bga