import torch
import triton
import triton.language as tl


def pattern(in_3, in_4):
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_3, in_4):
    return (in_3, in_4)


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
def fused_interp_sigmoid_mul_kernel(
    in4_ptr, in3_ptr, out_ptr,
    H_in, W_in, H_out, W_out,
    stride_in4_b, stride_in4_c, stride_in4_h, stride_in4_w,
    stride_in3_b, stride_in3_c, stride_in3_h, stride_in3_w,
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
    base_in4 = in4_ptr + b * stride_in4_b + c * stride_in4_c
    base_in3 = in3_ptr + b * stride_in3_b + c * stride_in3_c
    base_out = out_ptr + b * stride_out_b + c * stride_out_c

    # Load 4 values from in_4 for bilinear interpolation (gather loads)
    v00 = tl.load(base_in4 + y0 * stride_in4_h + x0 * stride_in4_w, mask=mask_w, other=0.0).to(tl.float32)
    v01 = tl.load(base_in4 + y0 * stride_in4_h + x1 * stride_in4_w, mask=mask_w, other=0.0).to(tl.float32)
    v10 = tl.load(base_in4 + y1 * stride_in4_h + x0 * stride_in4_w, mask=mask_w, other=0.0).to(tl.float32)
    v11 = tl.load(base_in4 + y1 * stride_in4_h + x1 * stride_in4_w, mask=mask_w, other=0.0).to(tl.float32)

    # Bilinear interpolation (fy scalar, fx vector -> broadcasts correctly)
    interp = (1.0 - fy) * (1.0 - fx) * v00 + (1.0 - fy) * fx * v01 + fy * (1.0 - fx) * v10 + fy * fx * v11

    # Sigmoid
    sig = tl.sigmoid(interp)

    # Load in_3 and multiply
    in3_val = tl.load(base_in3 + h_out * stride_in3_h + w_out * stride_in3_w, mask=mask_w, other=0.0).to(tl.float32)

    result = in3_val * sig

    # Store (auto-cast to output dtype)
    tl.store(base_out + h_out * stride_out_h + w_out * stride_out_w, result, mask=mask_w)


@torch.fx.wrap
def fused_interp_sigmoid_mul(in_3, in_4):
    B, C, H_out, W_out = in_3.shape
    H_in = in_4.shape[2]
    W_in = in_4.shape[3]

    out = torch.empty_like(in_3)

    grid = (B * C, H_out)

    fused_interp_sigmoid_mul_kernel[grid](
        in_4, in_3, out,
        H_in, W_in, H_out, W_out,
        in_4.stride()[0], in_4.stride()[1], in_4.stride()[2], in_4.stride()[3],
        in_3.stride()[0], in_3.stride()[1], in_3.stride()[2], in_3.stride()[3],
        out.stride()[0], out.stride()[1], out.stride()[2], out.stride()[3],
        B, C,
        BLOCK_W=64,
    )

    return out


def replacement_func():
    return fused_interp_sigmoid_mul