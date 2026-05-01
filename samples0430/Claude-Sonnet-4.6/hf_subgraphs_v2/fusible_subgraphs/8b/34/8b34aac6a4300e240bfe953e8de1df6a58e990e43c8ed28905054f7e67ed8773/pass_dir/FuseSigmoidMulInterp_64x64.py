import torch
import triton
import triton.language as tl


def pattern(conv2d_out, in_2):
    # pattern() is exempt from API validation - call wrap here so FX tracer
    # creates a single call_function(interpolate) leaf node, matching the target.
    torch.fx.wrap(torch.nn.functional.interpolate)
    tmp_6 = torch.sigmoid(conv2d_out)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    return tmp_8


def replacement_args(conv2d_out, in_2):
    return (conv2d_out, in_2)


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
def sigmoid_mul_interp_kernel(
    x_ptr, y_ptr, out_ptr,
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
    Fused: out = bilinear_upsample(in_2 * sigmoid(conv2d_out))
    Input x (conv2d_out): [B, C, H_IN,  W_IN]
    Input y (in_2):       [B, C, H_IN,  W_IN]
    Output:               [B, C, H_OUT, W_OUT]
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

    in_sp  = H_IN  * W_IN
    out_sp = H_OUT * W_OUT
    bc_in  = (b_idx * C + c_idx) * in_sp
    bc_out = (b_idx * C + c_idx) * out_sp

    # Load sigmoid(conv2d_out) at 4 neighbors
    sx_00 = tl.sigmoid(tl.load(x_ptr + bc_in + ih0 * W_IN + iw0, mask=mask).to(tl.float32))
    sx_01 = tl.sigmoid(tl.load(x_ptr + bc_in + ih0 * W_IN + iw1, mask=mask).to(tl.float32))
    sx_10 = tl.sigmoid(tl.load(x_ptr + bc_in + ih1 * W_IN + iw0, mask=mask).to(tl.float32))
    sx_11 = tl.sigmoid(tl.load(x_ptr + bc_in + ih1 * W_IN + iw1, mask=mask).to(tl.float32))

    # Load in_2 at 4 neighbors
    y_00 = tl.load(y_ptr + bc_in + ih0 * W_IN + iw0, mask=mask).to(tl.float32)
    y_01 = tl.load(y_ptr + bc_in + ih0 * W_IN + iw1, mask=mask).to(tl.float32)
    y_10 = tl.load(y_ptr + bc_in + ih1 * W_IN + iw0, mask=mask).to(tl.float32)
    y_11 = tl.load(y_ptr + bc_in + ih1 * W_IN + iw1, mask=mask).to(tl.float32)

    # Bilinear blend of (in_2 * sigmoid(conv2d_out))
    result = (w00 * (y_00 * sx_00) + w01 * (y_01 * sx_01) +
              w10 * (y_10 * sx_10) + w11 * (y_11 * sx_11))

    out_idx = bc_out + oh * W_OUT + ow
    if DTYPE_IS_FP16:
        tl.store(out_ptr + out_idx, result.to(tl.float16), mask=mask)
    elif DTYPE_IS_BF16:
        tl.store(out_ptr + out_idx, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def triton_sigmoid_mul_interp(conv2d_out, in_2):
    B, C, H_in, W_in = conv2d_out.shape
    H_out, W_out = 64, 64
    out = torch.empty((B, C, H_out, W_out), device=conv2d_out.device, dtype=in_2.dtype)
    n_elements = B * C * H_out * W_out
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    dtype   = in_2.dtype
    is_fp16 = (dtype == torch.float16)
    is_bf16 = (dtype == torch.bfloat16)
    sigmoid_mul_interp_kernel[grid](
        conv2d_out, in_2, out,
        B, C,
        H_IN=H_in, W_IN=W_in,
        H_OUT=H_out, W_OUT=W_out,
        DTYPE_IS_FP16=is_fp16,
        DTYPE_IS_BF16=is_bf16,
    )
    return out


def replacement_func():
    return triton_sigmoid_mul_interp