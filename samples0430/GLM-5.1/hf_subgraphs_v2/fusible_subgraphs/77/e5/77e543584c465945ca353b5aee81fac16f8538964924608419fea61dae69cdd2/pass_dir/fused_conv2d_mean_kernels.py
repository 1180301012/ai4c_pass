import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 4}, num_warps=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 8}, num_warps=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 4}, num_warps=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=8),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 8}, num_warps=8),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=8),
    ],
    key=['H_out', 'W_out', 'stride_h', 'stride_w'],
)
@triton.jit
def fused_conv2d_mean_partial_kernel(
    input_ptr, weight_ptr, conv_out_ptr, sum_buf_ptr,
    N, C, H, W, H_out, W_out,
    stride_h, stride_w,
    in_n_stride, in_c_stride, in_h_stride, in_w_stride,
    out_n_stride, out_c_stride, out_h_stride, out_w_stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # 3D grid: (N*C, H_out_tiles, W_out_tiles)
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    oh_offsets = oh_start + tl.arange(0, BLOCK_H)
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)

    oh_mask = oh_offsets < H_out
    ow_mask = ow_offsets < W_out

    # Load 3x3 weight for this channel
    # weight shape: [C, 1, 3, 3], contiguous => weight[c,0,kh,kw] at c*9 + kh*3 + kw
    w_base = c * 9
    w_offsets = tl.arange(0, 9)
    w_vals = tl.load(weight_ptr + w_base + w_offsets).to(tl.float32)

    # Initialize conv output accumulator in float32
    conv_vals = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    # For each (kh, kw) in the 3x3 kernel
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            # Input row/col indices with padding=1
            ih_offsets = oh_offsets * stride_h + kh - 1
            iw_offsets = ow_offsets * stride_w + kw - 1

            ih_valid = (ih_offsets >= 0) & (ih_offsets < H)
            iw_valid = (iw_offsets >= 0) & (iw_offsets < W)

            # 2D mask for valid positions
            mask = oh_mask[:, None] & ow_mask[None, :] & ih_valid[:, None] & iw_valid[None, :]

            # Load input values
            input_offsets = (
                n * in_n_stride + c * in_c_stride
                + ih_offsets[:, None] * in_h_stride + iw_offsets[None, :] * in_w_stride
            )
            input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)

            # Accumulate conv output
            w_val = w_vals[kh * 3 + kw]
            conv_vals = conv_vals + input_vals * w_val

    # Store conv output
    out_offsets = (
        n * out_n_stride + c * out_c_stride
        + oh_offsets[:, None] * out_h_stride + ow_offsets[None, :] * out_w_stride
    )
    out_mask = oh_mask[:, None] & ow_mask[None, :]
    tl.store(conv_out_ptr + out_offsets, conv_vals, mask=out_mask)

    # Accumulate partial sum for mean using atomic_add
    tile_sum = tl.sum(conv_vals * out_mask.to(tl.float32))
    # sum_buf shape: [N, C], float32, contiguous
    tl.atomic_add(sum_buf_ptr + pid_nc, tile_sum)


@triton.jit
def compute_mean_kernel(
    sum_buf_ptr, mean_out_ptr,
    N, C, spatial_size,
    mean_n_stride, mean_c_stride,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    sum_val = tl.load(sum_buf_ptr + pid)
    mean_val = sum_val / spatial_size

    # mean_out shape: [N, C, 1, 1]
    tl.store(mean_out_ptr + n * mean_n_stride + c * mean_c_stride, mean_val)


def fused_conv2d_mean_impl(input_tensor, weight_tensor, stride_h, stride_w, groups):
    """Fused depthwise conv2d + mean implementation."""
    N, C_in, H, W = input_tensor.shape
    C_out = groups  # depthwise conv
    H_out = (H + 2 * 1 - 3) // stride_h + 1  # padding=1, kernel=3
    W_out = (W + 2 * 1 - 3) // stride_w + 1

    # Allocate output tensors
    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    mean_out = torch.empty((N, C_out, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)

    # Allocate float32 sum buffer for mean accumulation
    sum_buf = torch.zeros((N, C_out), dtype=torch.float32, device=input_tensor.device)

    # Compute strides for input tensor
    in_n_stride, in_c_stride, in_h_stride, in_w_stride = input_tensor.stride()
    # Compute strides for output tensor
    out_n_stride, out_c_stride, out_h_stride, out_w_stride = conv_out.stride()

    # Ensure weight is contiguous and on the right device
    weight_tensor = weight_tensor.contiguous().to(input_tensor.device)

    # Launch conv+partial_sum kernel with 3D grid
    total_nc = N * C_out
    grid = lambda kwargs: (
        total_nc,
        triton.cdiv(H_out, kwargs['BLOCK_H']),
        triton.cdiv(W_out, kwargs['BLOCK_W']),
    )

    fused_conv2d_mean_partial_kernel[grid](
        input_tensor, weight_tensor, conv_out, sum_buf,
        N, C_out, H, W, H_out, W_out,
        stride_h, stride_w,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride,
    )

    # Launch mean kernel
    compute_mean_kernel[(total_nc,)](
        sum_buf, mean_out,
        N, C_out, H_out * W_out,
        mean_out.stride()[0], mean_out.stride()[1],
    )

    return conv_out, mean_out


# Shared dispatch wrapper - all pass files import and return this same function object
@torch.fx.wrap
def fused_conv2d_mean_dispatch(weight, input, route_str):
    if route_str == "s1_g256":
        return fused_conv2d_mean_impl(input, weight, 1, 1, 256)
    elif route_str == "s2_g256":
        return fused_conv2d_mean_impl(input, weight, 2, 2, 256)
    elif route_str == "s1_g384":
        return fused_conv2d_mean_impl(input, weight, 1, 1, 384)
    elif route_str == "s2_g384":
        return fused_conv2d_mean_impl(input, weight, 2, 2, 384)
    elif route_str == "s1_g768":
        return fused_conv2d_mean_impl(input, weight, 1, 1, 768)
    else:
        raise ValueError(f"Unknown route: {route_str}")