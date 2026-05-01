import torch
import triton
import triton.language as tl


def pattern(in_4, in_3):
    # pattern() is exempt from API validation - call wrap here so FX tracer
    # creates a single call_function(interpolate) leaf node, matching the target.
    torch.fx.wrap(torch.nn.functional.interpolate)
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_4, in_3):
    return (in_4, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['B', 'C'],
)
@triton.jit
def interp_sigmoid_mul_kernel(
    in4_ptr, in3_ptr, out_ptr,
    B, C,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    DTYPE_IS_FP16: tl.constexpr,
    DTYPE_IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: out = in_3 * sigmoid(bilinear_upsample(in_4))
    Input in_4:  [B, C, H_IN,  W_IN]
    Input in_3:  [B, C, H_OUT, W_OUT]
    Output:      [B, C, H_OUT, W_OUT]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = B * C * H_OUT * W_OUT
    mask = offsets < n_elements

    # Decompose linear index -> (b, c, oh, ow)
    ow    = offsets % W_OUT
    oh    = (offsets // W_OUT) % H_OUT
    c_idx = (offsets // (H_OUT * W_OUT)) % C
    b_idx = offsets // (C * H_OUT * W_OUT)

    # Bilinear source coords (align_corners=False)
    SCALE_H: tl.constexpr = H_IN / H_OUT
    SCALE_W: tl.constexpr = W_IN / W_OUT

    src_h = (tl.cast(oh, tl.float32) + 0.5) * SCALE_H - 0.5
    src_w = (tl.cast(ow, tl.float32) + 0.5) * SCALE_W - 0.5

    floor_h_f = tl.math.floor(src_h)
    floor_w_f = tl.math.floor(src_w)
    floor_h   = tl.cast(floor_h_f, tl.int32)
    floor_w   = tl.cast(floor_w_f, tl.int32)

    ih0 = tl.maximum(floor_h,     0)
    ih1 = tl.minimum(floor_h + 1, H_IN - 1)
    iw0 = tl.maximum(floor_w,     0)
    iw1 = tl.minimum(floor_w + 1, W_IN - 1)

    wh  = src_h - floor_h_f
    ww  = src_w - floor_w_f
    w00 = (1.0 - wh) * (1.0 - ww)
    w01 = (1.0 - wh) * ww
    w10 = wh * (1.0 - ww)
    w11 = wh * ww

    in_sp   = H_IN  * W_IN
    out_sp  = H_OUT * W_OUT
    bc_in   = (b_idx * C + c_idx) * in_sp
    bc_out  = (b_idx * C + c_idx) * out_sp

    # Load 4 in_4 neighbors and bilinear blend
    v00 = tl.load(in4_ptr + bc_in + ih0 * W_IN + iw0, mask=mask).to(tl.float32)
    v01 = tl.load(in4_ptr + bc_in + ih0 * W_IN + iw1, mask=mask).to(tl.float32)
    v10 = tl.load(in4_ptr + bc_in + ih1 * W_IN + iw0, mask=mask).to(tl.float32)
    v11 = tl.load(in4_ptr + bc_in + ih1 * W_IN + iw1, mask=mask).to(tl.float32)
    blended = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11

    # Load in_3 and compute final
    in3_val = tl.load(in3_ptr + bc_out + oh * W_OUT + ow, mask=mask).to(tl.float32)
    result  = in3_val * tl.sigmoid(blended)

    out_idx = bc_out + oh * W_OUT + ow
    if DTYPE_IS_FP16:
        tl.store(out_ptr + out_idx, result.to(tl.float16), mask=mask)
    elif DTYPE_IS_BF16:
        tl.store(out_ptr + out_idx, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def triton_interp_sigmoid_mul(in_4, in_3):
    B, C, H_in, W_in = in_4.shape
    H_out, W_out = 64, 64
    out = torch.empty((B, C, H_out, W_out), device=in_4.device, dtype=in_3.dtype)
    n_elements = B * C * H_out * W_out
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    dtype   = in_3.dtype
    is_fp16 = (dtype == torch.float16)
    is_bf16 = (dtype == torch.bfloat16)
    interp_sigmoid_mul_kernel[grid](
        in_4, in_3, out,
        B, C,
        H_IN=H_in, W_IN=W_in,
        H_OUT=H_out, W_OUT=W_out,
        DTYPE_IS_FP16=is_fp16,
        DTYPE_IS_BF16=is_bf16,
    )
    return out


def replacement_func():
    return triton_interp_sigmoid_mul