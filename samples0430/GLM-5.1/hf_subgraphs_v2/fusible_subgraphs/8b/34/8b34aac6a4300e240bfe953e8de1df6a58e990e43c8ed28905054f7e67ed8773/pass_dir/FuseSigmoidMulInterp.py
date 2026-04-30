import torch
import triton
import triton.language as tl


def pattern(in_2, conv_result):
    tmp_6 = torch.sigmoid(conv_result)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    return tmp_8


def replacement_args(in_2, conv_result):
    return (in_2, conv_result)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_W': 64}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_W': 64}, num_warps=16, num_stages=2),
    ],
    key=['H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def fused_sigmoid_mul_interp_kernel(
    conv_ptr, in2_ptr, out_ptr,
    H_in, W_in, H_out, W_out,
    stride_conv_b, stride_conv_c, stride_conv_h, stride_conv_w,
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    B, C,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_bc // C
    c = pid_bc % C

    h_out = pid_h
    w_out = tl.arange(0, BLOCK_W)
    mask_w = w_out < W_out

    # Source y coordinate (scalar for this row)
    src_y = (h_out + 0.5) * H_in / H_out - 0.5
    src_y = tl.maximum(0.0, tl.minimum(float(H_in - 1), src_y))

    y0 = tl.floor(src_y).to(tl.int32)
    y1 = tl.minimum(y0 + 1, H_in - 1)
    fy = src_y - y0.to(tl.float32)

    # Source x coordinates (vector across columns)
    src_x = (w_out + 0.5) * W_in / W_out - 0.5
    src_x = tl.where(mask_w, tl.maximum(0.0, tl.minimum(float(W_in - 1), src_x)), 0.0)

    x0 = tl.floor(src_x).to(tl.int32)
    x1 = tl.minimum(x0 + 1, W_in - 1)
    fx = src_x - x0.to(tl.float32)

    # Base offsets for this (b, c) pair
    base_conv = conv_ptr + b * stride_conv_b + c * stride_conv_c
    base_in2 = in2_ptr + b * stride_in2_b + c * stride_in2_c
    base_out = out_ptr + b * stride_out_b + c * stride_out_c

    # Load conv values at 4 corners
    conv_00 = tl.load(base_conv + y0 * stride_conv_h + x0 * stride_conv_w, mask=mask_w, other=0.0).to(tl.float32)
    conv_01 = tl.load(base_conv + y0 * stride_conv_h + x1 * stride_conv_w, mask=mask_w, other=0.0).to(tl.float32)
    conv_10 = tl.load(base_conv + y1 * stride_conv_h + x0 * stride_conv_w, mask=mask_w, other=0.0).to(tl.float32)
    conv_11 = tl.load(base_conv + y1 * stride_conv_h + x1 * stride_conv_w, mask=mask_w, other=0.0).to(tl.float32)

    # Load in_2 values at 4 corners
    in2_00 = tl.load(base_in2 + y0 * stride_in2_h + x0 * stride_in2_w, mask=mask_w, other=0.0).to(tl.float32)
    in2_01 = tl.load(base_in2 + y0 * stride_in2_h + x1 * stride_in2_w, mask=mask_w, other=0.0).to(tl.float32)
    in2_10 = tl.load(base_in2 + y1 * stride_in2_h + x0 * stride_in2_w, mask=mask_w, other=0.0).to(tl.float32)
    in2_11 = tl.load(base_in2 + y1 * stride_in2_h + x1 * stride_in2_w, mask=mask_w, other=0.0).to(tl.float32)

    # Sigmoid(conv) * in_2 at each corner
    f00 = in2_00 * tl.sigmoid(conv_00)
    f01 = in2_01 * tl.sigmoid(conv_01)
    f10 = in2_10 * tl.sigmoid(conv_10)
    f11 = in2_11 * tl.sigmoid(conv_11)

    # Bilinear interpolation of f values
    # fy scalar, fx vector -> broadcasts correctly
    result = (1.0 - fy) * (1.0 - fx) * f00 + (1.0 - fy) * fx * f01 + fy * (1.0 - fx) * f10 + fy * fx * f11

    # Store (auto-cast to output dtype)
    tl.store(base_out + h_out * stride_out_h + w_out * stride_out_w, result, mask=mask_w)


@torch.fx.wrap
def fused_sigmoid_mul_interp(in_2, conv_result):
    B, C, H_in, W_in = conv_result.shape
    H_out = 64
    W_out = 64

    out = torch.empty(B, C, H_out, W_out, dtype=conv_result.dtype, device=conv_result.device)

    grid = (B * C, H_out)

    fused_sigmoid_mul_interp_kernel[grid](
        conv_result, in_2, out,
        H_in, W_in, H_out, W_out,
        conv_result.stride()[0], conv_result.stride()[1], conv_result.stride()[2], conv_result.stride()[3],
        in_2.stride()[0], in_2.stride()[1], in_2.stride()[2], in_2.stride()[3],
        out.stride()[0], out.stride()[1], out.stride()[2], out.stride()[3],
        B, C,
        BLOCK_W=64,
    )

    return out


def replacement_func():
    return fused_sigmoid_mul_interp