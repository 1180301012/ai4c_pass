import torch
import triton
import triton.language as tl


def pattern(in_0, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = tmp_2 + tmp_7
    return tmp_8


def replacement_args(in_0, in_2):
    return (in_0, in_2)


@triton.jit
def fused_star_relu_kernel(
    in_2_ptr, in_0_ptr, out_ptr,
    B, C, H, W,
    stride_in2_n, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_in0_c,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    DTYPE_OUT: tl.constexpr,
):
    pid = tl.program_id(0)

    num_h_tiles = tl.cdiv(H, BLOCK_H)
    num_w_tiles = tl.cdiv(W, BLOCK_W)
    num_spatial_tiles = num_h_tiles * num_w_tiles

    bc_idx = pid // num_spatial_tiles
    spatial_idx = pid % num_spatial_tiles

    n = bc_idx // C
    c = bc_idx % C

    h_tile = spatial_idx // num_w_tiles
    w_tile = spatial_idx % num_w_tiles

    h_start = h_tile * BLOCK_H
    w_start = w_tile * BLOCK_W

    # Output position offsets
    h_out = h_start + tl.arange(0, BLOCK_H)
    w_out = w_start + tl.arange(0, BLOCK_W)

    # Padded position offsets (for 3x3 stencil, need -1 to +1 neighborhood)
    h_pad = h_start - 1 + tl.arange(0, BLOCK_H + 2)
    w_pad = w_start - 1 + tl.arange(0, BLOCK_W + 2)

    # Masks for valid output positions
    h_out_mask = h_out < H
    w_out_mask = w_out < W
    out_mask = h_out_mask[:, None] & w_out_mask[None, :]  # [BLOCK_H, BLOCK_W]

    # Masks for valid padded positions
    h_pad_mask = (h_pad >= 0) & (h_pad < H)
    w_pad_mask = (w_pad >= 0) & (w_pad < W)
    pad_mask_2d = h_pad_mask[:, None] & w_pad_mask[None, :]  # [BLOCK_H+2, BLOCK_W+2]

    # Base pointer for this (n, c) pair
    base_ptr = in_2_ptr + n * stride_in2_n + c * stride_in2_c

    # 2D offsets for padded block
    pad_offsets = h_pad[:, None] * stride_in2_h + w_pad[None, :] * stride_in2_w

    # Load padded block with zero-padding for out-of-bounds, then upcast to float32
    padded = tl.load(base_ptr + pad_offsets, mask=pad_mask_2d, other=0.0).to(tl.float32)

    # Apply ReLU in float32
    relu_padded = tl.where(padded > 0.0, padded, 0.0)

    # Compute 3x3 avg_pool using shifted views of the padded block
    # For output position (i, j), the 3x3 window covers padded[i:i+3, j:j+3]
    # i ranges 0..BLOCK_H-1, j ranges 0..BLOCK_W-1
    r00 = relu_padded[0:BLOCK_H, 0:BLOCK_W]
    r01 = relu_padded[0:BLOCK_H, 1:BLOCK_W + 1]
    r02 = relu_padded[0:BLOCK_H, 2:BLOCK_W + 2]
    r10 = relu_padded[1:BLOCK_H + 1, 0:BLOCK_W]
    r11 = relu_padded[1:BLOCK_H + 1, 1:BLOCK_W + 1]  # center = relu of original
    r12 = relu_padded[1:BLOCK_H + 1, 2:BLOCK_W + 2]
    r20 = relu_padded[2:BLOCK_H + 2, 0:BLOCK_W]
    r21 = relu_padded[2:BLOCK_H + 2, 1:BLOCK_W + 1]
    r22 = relu_padded[2:BLOCK_H + 2, 2:BLOCK_W + 2]

    # Sum of 3x3 window of ReLU values
    sum_3x3 = r00 + r01 + r02 + r10 + r11 + r12 + r20 + r21 + r22

    # Count valid elements for count_include_pad=False
    m00 = pad_mask_2d[0:BLOCK_H, 0:BLOCK_W].to(tl.float32)
    m01 = pad_mask_2d[0:BLOCK_H, 1:BLOCK_W + 1].to(tl.float32)
    m02 = pad_mask_2d[0:BLOCK_H, 2:BLOCK_W + 2].to(tl.float32)
    m10 = pad_mask_2d[1:BLOCK_H + 1, 0:BLOCK_W].to(tl.float32)
    m11 = pad_mask_2d[1:BLOCK_H + 1, 1:BLOCK_W + 1].to(tl.float32)
    m12 = pad_mask_2d[1:BLOCK_H + 1, 2:BLOCK_W + 2].to(tl.float32)
    m20 = pad_mask_2d[2:BLOCK_H + 2, 0:BLOCK_W].to(tl.float32)
    m21 = pad_mask_2d[2:BLOCK_H + 2, 1:BLOCK_W + 1].to(tl.float32)
    m22 = pad_mask_2d[2:BLOCK_H + 2, 2:BLOCK_W + 2].to(tl.float32)

    count_3x3 = m00 + m01 + m02 + m10 + m11 + m12 + m20 + m21 + m22

    # avg_pool = sum / count (count_include_pad=False)
    avg_pool = sum_3x3 / count_3x3

    # relu of the original value at the output position
    relu_center = r11

    # subtract = avg_pool - relu_center
    subtract = avg_pool - relu_center

    # Load layer_scale_1 for this channel and upcast to float32
    ls1 = tl.load(in_0_ptr + c * stride_in0_c).to(tl.float32)

    # result = relu_center + ls1 * subtract
    result = relu_center + ls1 * subtract

    # Convert back to output dtype
    result_out = result.to(DTYPE_OUT)

    # Store result
    out_base_ptr = out_ptr + n * stride_out_n + c * stride_out_c
    out_offsets = h_out[:, None] * stride_out_h + w_out[None, :] * stride_out_w
    tl.store(out_base_ptr + out_offsets, result_out, mask=out_mask)


@torch.fx.wrap
def fused_star_relu(in_0, in_2):
    B, C, H, W = in_2.shape
    out = torch.empty_like(in_2)

    # Determine output dtype for constexpr
    if in_2.dtype == torch.float32:
        DTYPE_OUT = tl.float32
    elif in_2.dtype == torch.float16:
        DTYPE_OUT = tl.float16
    elif in_2.dtype == torch.bfloat16:
        DTYPE_OUT = tl.bfloat16
    else:
        DTYPE_OUT = tl.float32

    BLOCK_H = 8
    BLOCK_W = 8

    num_h_tiles = (H + BLOCK_H - 1) // BLOCK_H
    num_w_tiles = (W + BLOCK_W - 1) // BLOCK_W
    num_spatial_tiles = num_h_tiles * num_w_tiles
    grid = (B * C * num_spatial_tiles,)

    fused_star_relu_kernel[grid](
        in_2_ptr=in_2,
        in_0_ptr=in_0,
        out_ptr=out,
        B=B, C=C, H=H, W=W,
        stride_in2_n=in_2.stride(0),
        stride_in2_c=in_2.stride(1),
        stride_in2_h=in_2.stride(2),
        stride_in2_w=in_2.stride(3),
        stride_in0_c=in_0.stride(0),
        stride_out_n=out.stride(0),
        stride_out_c=out.stride(1),
        stride_out_h=out.stride(2),
        stride_out_w=out.stride(3),
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        DTYPE_OUT=DTYPE_OUT,
    )

    return out


def replacement_func():
    return fused_star_relu