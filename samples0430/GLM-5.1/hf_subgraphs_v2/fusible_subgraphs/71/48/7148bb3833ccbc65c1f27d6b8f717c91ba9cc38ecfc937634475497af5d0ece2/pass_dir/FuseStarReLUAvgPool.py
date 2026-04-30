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
def fused_star_relu_avg_pool_kernel(
    input_ptr, scale_ptr, output_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc - (pid_nc // C) * C

    if n >= N:
        return

    # Load scale for this channel
    scale_val = tl.load(scale_ptr + c).to(tl.float32)

    # Output spatial positions
    h_out = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    h_out_mask = h_out < H
    w_out_mask = w_out < W
    out_mask = h_out_mask[:, None] & w_out_mask[None, :]

    base = input_ptr + n * stride_n + c * stride_c

    # Load extended input tile: (BLOCK_H+2) x (BLOCK_W+2)
    # This covers all 3x3 neighborhoods for the output block
    h_ext = pid_h * BLOCK_H - 1 + tl.arange(0, BLOCK_H + 2)
    w_ext = pid_w * BLOCK_W - 1 + tl.arange(0, BLOCK_W + 2)

    h_ext_valid = (h_ext >= 0) & (h_ext < H)
    w_ext_valid = (w_ext >= 0) & (w_ext < W)
    ext_mask = h_ext_valid[:, None] & w_ext_valid[None, :]

    # Load the extended tile
    ext_tile = tl.load(
        base + h_ext[:, None] * stride_h + w_ext[None, :] * stride_w,
        mask=ext_mask,
        other=0.0,
    ).to(tl.float32)

    # Apply relu to entire tile
    relu_tile = tl.maximum(ext_tile, 0.0)

    # Compute avg_pool for each output position by iterating over 3x3 neighborhood
    # For output position (i,j), avg_pool = average of relu_tile[i+dh, j+dw]
    # for dh,dw in {0,1,2}, only counting valid (non-padding) positions
    avg_pool_sum = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    neighbor_count = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    for dh in tl.static_range(3):
        for dw in tl.static_range(3):
            # Extract sub-tile from the extended relu tile
            # relu_tile[dh:dh+BLOCK_H, dw:dw+BLOCK_W] gives neighbor values
            # for all output positions at this (dh,dw) offset
            sub_relu = relu_tile[dh:dh + BLOCK_H, dw:dw + BLOCK_W]

            # Validity of this neighbor for each output position
            # h_ext_valid[dh:dh+BLOCK_H] tells which rows of this sub-tile are valid
            # w_ext_valid[dw:dw+BLOCK_W] tells which columns are valid
            sub_h_valid = h_ext_valid[dh:dh + BLOCK_H]
            sub_w_valid = w_ext_valid[dw:dw + BLOCK_W]
            sub_valid = sub_h_valid[:, None] & sub_w_valid[None, :] & out_mask

            # Add valid neighbor values to sum and increment count
            avg_pool_sum += sub_relu * sub_valid.to(tl.float32)
            neighbor_count += sub_valid.to(tl.float32)

    # Avoid division by zero for invalid output positions
    neighbor_count_safe = tl.maximum(neighbor_count, 1.0)
    avg_pool_val = avg_pool_sum / neighbor_count_safe

    # Extract center relu values: relu_tile[1:BLOCK_H+1, 1:BLOCK_W+1]
    relu_center = relu_tile[1:BLOCK_H + 1, 1:BLOCK_W + 1]

    # Compute output: relu_center + scale * (avg_pool_val - relu_center)
    out_val = relu_center + scale_val * (avg_pool_val - relu_center)

    # Store output
    tl.store(
        output_ptr + n * stride_n + c * stride_c + h_out[:, None] * stride_h + w_out[None, :] * stride_w,
        out_val,
        mask=out_mask,
    )


@torch.fx.wrap
def fused_star_relu_avg_pool(in_0, in_2):
    N, C, H, W = in_2.shape
    stride_n, stride_c, stride_h, stride_w = in_2.stride()

    output = torch.empty_like(in_2)

    BLOCK_H = 8
    BLOCK_W = 8

    grid_h = (H + BLOCK_H - 1) // BLOCK_H
    grid_w = (W + BLOCK_W - 1) // BLOCK_W

    grid = (N * C, grid_h, grid_w)

    fused_star_relu_avg_pool_kernel[grid](
        input_ptr=in_2,
        scale_ptr=in_0,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        stride_n=stride_n, stride_c=stride_c, stride_h=stride_h, stride_w=stride_w,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )

    return output


def replacement_func():
    return fused_star_relu_avg_pool