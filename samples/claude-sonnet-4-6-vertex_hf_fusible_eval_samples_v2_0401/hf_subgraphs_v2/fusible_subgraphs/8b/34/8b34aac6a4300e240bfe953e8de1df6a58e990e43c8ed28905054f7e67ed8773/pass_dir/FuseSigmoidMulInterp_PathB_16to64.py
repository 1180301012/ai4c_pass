import torch
import triton
import triton.language as tl


# Path B: sigmoid(conv2d) -> multiply(in_2) -> bilinear-upsample(16->64)
def pattern(conv2d_out, in_2):
    tmp_6 = torch.sigmoid(conv2d_out)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    return tmp_8


def replacement_args(conv2d_out, in_2):
    return (conv2d_out, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def sigmoid_mul_interp_kernel(
    conv2d_ptr,  # (B, C, H_in, W_in) = (B, 128, 16, 16)
    in2_ptr,     # (B, C, H_in, W_in) = (B, 128, 16, 16)
    out_ptr,     # (B, C, H_out, W_out) = (B, 128, 64, 64)
    B, C,
    H_in, W_in,
    H_out, W_out,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < total

    # decode NCHW output indices
    w_out  = (offsets % W_out).to(tl.int32)
    tmp_   = offsets // W_out
    h_out  = (tmp_ % H_out).to(tl.int32)
    tmp_   = tmp_ // H_out
    c      = (tmp_ % C).to(tl.int32)
    b      = (tmp_ // C).to(tl.int32)

    # bilinear coords (align_corners=False): src = (dst+0.5)*(H_in/H_out) - 0.5
    scale  = H_in.to(tl.float32) / H_out.to(tl.float32)
    x_in   = (w_out.to(tl.float32) + 0.5) * scale - 0.5
    y_in   = (h_out.to(tl.float32) + 0.5) * scale - 0.5

    x0_f   = tl.floor(x_in)
    y0_f   = tl.floor(y_in)
    x0     = x0_f.to(tl.int32)
    y0     = y0_f.to(tl.int32)
    x0c    = tl.maximum(x0,     0)
    y0c    = tl.maximum(y0,     0)
    x1c    = tl.minimum(x0 + 1, W_in - 1)
    y1c    = tl.minimum(y0 + 1, H_in - 1)

    fx  = x_in - x0_f
    fy  = y_in - y0_f
    w00 = (1.0 - fx) * (1.0 - fy)
    w01 = fx          * (1.0 - fy)
    w10 = (1.0 - fx) * fy
    w11 = fx          * fy

    base_in  = (b * C + c) * (H_in  * W_in)
    base_out = (b * C + c) * (H_out * W_out)

    # load conv2d at 4 neighbors, compute sigmoid * in2, then bilinear combine
    c00 = tl.load(conv2d_ptr + base_in + y0c * W_in + x0c, mask=mask, other=0.0).to(tl.float32)
    c01 = tl.load(conv2d_ptr + base_in + y0c * W_in + x1c, mask=mask, other=0.0).to(tl.float32)
    c10 = tl.load(conv2d_ptr + base_in + y1c * W_in + x0c, mask=mask, other=0.0).to(tl.float32)
    c11 = tl.load(conv2d_ptr + base_in + y1c * W_in + x1c, mask=mask, other=0.0).to(tl.float32)

    a00 = tl.load(in2_ptr + base_in + y0c * W_in + x0c, mask=mask, other=0.0).to(tl.float32)
    a01 = tl.load(in2_ptr + base_in + y0c * W_in + x1c, mask=mask, other=0.0).to(tl.float32)
    a10 = tl.load(in2_ptr + base_in + y1c * W_in + x0c, mask=mask, other=0.0).to(tl.float32)
    a11 = tl.load(in2_ptr + base_in + y1c * W_in + x1c, mask=mask, other=0.0).to(tl.float32)

    # in2 * sigmoid(conv2d) at each neighbor, then bilinear blend
    val_00 = a00 * tl.sigmoid(c00)
    val_01 = a01 * tl.sigmoid(c01)
    val_10 = a10 * tl.sigmoid(c10)
    val_11 = a11 * tl.sigmoid(c11)

    result = (val_00 * w00 + val_01 * w01 + val_10 * w10 + val_11 * w11).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + base_out + h_out * W_out + w_out, result, mask=mask)


@torch.fx.wrap
def sigmoid_mul_interp(conv2d_out, in_2):
    B, C, H_in, W_in = conv2d_out.shape
    H_out, W_out = 64, 64
    out   = torch.empty((B, C, H_out, W_out), dtype=in_2.dtype, device=in_2.device)
    total = B * C * H_out * W_out
    grid  = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    sigmoid_mul_interp_kernel[grid](
        conv2d_out, in_2, out,
        B, C,
        H_in, W_in,
        H_out, W_out,
        total,
    )
    return out


def replacement_func():
    return sigmoid_mul_interp