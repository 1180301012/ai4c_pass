import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    C, H, W, total_hw,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    hw_start = pid * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < total_hw

    # Decode flat offset to (b, h, w)
    HW_val = H * W
    b = hw_offsets // HW_val
    hw_in_b = hw_offsets % HW_val
    h = hw_in_b // W
    w = hw_in_b % W

    # Base offset for each spatial position (without channel contribution)
    # input[b, c, h, w] = b * (C * HW_val) + c * HW_val + h * W + w
    CHW_val = C * HW_val
    base_off = b * CHW_val + h * W + w  # 1D (BLOCK_HW)

    # Load bias and init result in float32
    bias_val = tl.load(bias_ptr).to(tl.float32)
    result = bias_val + tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Reduce over channels
    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C

        # Load weight - 1D (BLOCK_C), weight[c] at flat offset c
        # weight shape is [1, C, 1, 1], contiguous: weight[0,c,0,0] = c
        weight_vals = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        # Load input - 2D (BLOCK_C, BLOCK_HW) for better memory coalescing
        # input_offs[c, hw] = c_offsets[c] * HW_val + base_off[hw]
        input_offs = c_offsets[:, None] * HW_val + base_off[None, :]
        input_mask_2d = c_mask[:, None] & hw_mask[None, :]
        input_vals = tl.load(input_ptr + input_offs, mask=input_mask_2d, other=0.0).to(tl.float32)

        # Dot product: weight_vals[:, None] * input_vals -> (BLOCK_C, BLOCK_HW)
        # Sum over axis 0 (channels) -> (BLOCK_HW)
        result += tl.sum(weight_vals[:, None] * input_vals, axis=0)

    # Store output at flat hw_offsets
    tl.store(output_ptr + hw_offsets, result, mask=hw_mask)


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * HW

    # Pass 1: Find max
    max_val = float('-inf')
    for start in range(0, HW, BLOCK_SIZE):
        offsets = row_start + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + HW
        row = tl.load(input_ptr + offsets, mask=mask, other=float('-inf')).to(tl.float32)
        row = tl.where(mask, row, float('-inf'))
        chunk_max = tl.max(row, axis=0)
        max_val = tl.maximum(max_val, chunk_max)

    # Pass 2: Compute exp and sum
    sum_val = tl.zeros([1], dtype=tl.float32)
    for start in range(0, HW, BLOCK_SIZE):
        offsets = row_start + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + HW
        row = tl.load(input_ptr + offsets, mask=mask, other=float('-inf')).to(tl.float32)
        row = tl.where(mask, row, float('-inf'))
        exp_row = tl.exp(row - max_val)
        sum_val = sum_val + tl.sum(tl.where(mask, exp_row, 0.0), axis=0)

    # Pass 3: Normalize and store
    for start in range(0, HW, BLOCK_SIZE):
        offsets = row_start + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + HW
        row = tl.load(input_ptr + offsets, mask=mask, other=float('-inf')).to(tl.float32)
        row = tl.where(mask, row, float('-inf'))
        output = tl.exp(row - max_val) / sum_val
        output = tl.where(mask, output, 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)


def _fused_conv2d_view_softmax_impl(input, weight, bias):
    B = input.shape[0]
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    HW_num = H * W

    # Allocate conv2d output buffer (B, H, W) - will be input to softmax
    conv_output = torch.empty((B, H, W), dtype=input.dtype, device=input.device)

    # Allocate softmax output buffer (B, 1, HW_num)
    softmax_output = torch.empty((B, 1, HW_num), dtype=input.dtype, device=input.device)

    # Conv2d kernel launch
    total_hw = B * HW_num
    BLOCK_HW = 64
    BLOCK_C = 64
    num_conv_programs = (total_hw + BLOCK_HW - 1) // BLOCK_HW
    conv2d_1x1_kernel[(num_conv_programs,)](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias,
        output_ptr=conv_output,
        C=C, H=H, W=W, total_hw=total_hw,
        BLOCK_HW=BLOCK_HW, BLOCK_C=BLOCK_C,
    )

    # Softmax kernel launch
    BLOCK_SOFTMAX = 1024
    softmax_kernel[(B,)](
        input_ptr=conv_output, output_ptr=softmax_output,
        HW=HW_num,
        BLOCK_SIZE=BLOCK_SOFTMAX,
    )

    return softmax_output


@torch.fx.wrap
def _dispatch(input, weight, bias, route):
    # All routes call the same implementation - route is just for differentiation
    return _fused_conv2d_view_softmax_impl(input, weight, bias)