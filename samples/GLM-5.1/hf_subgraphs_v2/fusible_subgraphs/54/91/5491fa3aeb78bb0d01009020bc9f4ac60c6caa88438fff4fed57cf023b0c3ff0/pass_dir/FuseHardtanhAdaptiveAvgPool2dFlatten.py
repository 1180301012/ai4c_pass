import torch
import triton
import triton.language as tl

# Pattern matching - mirrors the exact computation in model.py
def pattern(in_0 : torch.Tensor):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(tmp_1.shape[0], -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return (tmp_3,)

def replacement_args(in_0 : torch.Tensor):
    return (in_0,)

@triton.jit
def hardtanh_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    inv_HW,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (batch_idx, channel_idx) pair
    bc_idx = tl.program_id(0)
    batch_idx = bc_idx // C
    channel_idx = bc_idx % C

    # Base offset in input tensor for this (batch, channel)
    base_offset = batch_idx * C * H * W + channel_idx * H * W

    # Accumulate sum over HW spatial dimensions
    acc = 0.0
    for block_start in range(0, HW, BLOCK_HW):
        offsets = block_start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        # Load input values
        vals = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        # Apply hardtanh: clamp to [0.0, 6.0]
        clamped = tl.minimum(tl.maximum(vals, 0.0), 6.0)
        # Accumulate
        acc += tl.sum(clamped, axis=0) if BLOCK_HW < HW else tl.sum(clamped * mask, axis=0)

    # Compute average
    avg = acc * inv_HW

    # Store to output[batch_idx, channel_idx]
    tl.store(output_ptr + batch_idx * C + channel_idx, avg)


@triton.jit
def hardtanh_avg_pool2d_kernel_dynamic(
    input_ptr,
    output_ptr,
    B,
    C,
    H,
    W,
    HW,
    inv_HW,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (batch_idx, channel_idx) pair
    bc_idx = tl.program_id(0)
    batch_idx = bc_idx // C
    channel_idx = bc_idx % C

    # Accumulate sum over HW spatial dimensions
    acc = 0.0
    num_valid = 0

    for h_start in range(0, H, 1):
        for w_start in range(0, W, BLOCK_HW):
            w_offsets = w_start + tl.arange(0, BLOCK_HW)
            w_mask = w_offsets < W

            # Compute full offsets: batch * stride_b + channel * stride_c + h * stride_h + w * stride_w
            offsets = batch_idx * stride_b + channel_idx * stride_c + h_start * stride_h + w_offsets * stride_w

            # Load input values
            vals = tl.load(input_ptr + offsets, mask=w_mask, other=0.0)
            # Apply hardtanh: clamp to [0.0, 6.0]
            clamped = tl.minimum(tl.maximum(vals, 0.0), 6.0)
            # Accumulate
            acc += tl.sum(clamped * w_mask, axis=0)
            num_valid += tl.sum(w_mask, axis=0)

    # Compute average
    avg = acc / num_valid

    # Store to output[batch_idx, channel_idx]
    tl.store(output_ptr + batch_idx * C + channel_idx, avg)


@triton.jit
def hardtanh_avg_pool2d_kernel_v2(
    input_ptr,
    output_ptr,
    B,
    C,
    H,
    W,
    HW,
    inv_HW,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch_idx, channel_idx) pair
    bc_idx = tl.program_id(0)
    batch_idx = bc_idx // C
    channel_idx = bc_idx % C

    # Accumulate sum over HW spatial dimensions
    acc = 0.0

    for block_start in range(0, HW, BLOCK_SIZE):
        offsets_in_hw = block_start + tl.arange(0, BLOCK_SIZE)
        hw_mask = offsets_in_hw < HW

        # Convert HW offset to h, w coordinates
        h_coord = offsets_in_hw // W
        w_coord = offsets_in_hw % W

        # Compute full offsets in input tensor
        input_offsets = batch_idx * stride_b + channel_idx * stride_c + h_coord * stride_h + w_coord * stride_w

        # Load input values
        vals = tl.load(input_ptr + input_offsets, mask=hw_mask, other=0.0)
        # Apply hardtanh: clamp to [0.0, 6.0]
        clamped = tl.minimum(tl.maximum(vals, 0.0), 6.0)
        # Accumulate
        acc += tl.sum(clamped * hw_mask, axis=0)

    # Compute average
    avg = acc * inv_HW

    # Store to output[batch_idx, channel_idx]
    tl.store(output_ptr + batch_idx * C + channel_idx, avg)


@torch.fx.wrap
def hardtanh_avg_pool2d_flatten(input_tensor):
    B, C, H, W = input_tensor.shape
    HW = H * W
    inv_HW = 1.0 / float(HW)

    output = torch.empty(B, C, dtype=input_tensor.dtype, device=input_tensor.device)

    strides = input_tensor.stride()
    stride_b = strides[0]
    stride_c = strides[1]
    stride_h = strides[2]
    stride_w = strides[3]

    num_programs = B * C

    # Choose block size based on HW size
    BLOCK_SIZE = max(32, min(HW, 4096))
    # Round to power of 2 for efficiency
    BLOCK_SIZE = 1 << (BLOCK_SIZE - 1).bit_length()
    if BLOCK_SIZE > HW:
        BLOCK_SIZE = HW if HW >= 32 else 32

    hardtanh_avg_pool2d_kernel_v2[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        B=B,
        C=C,
        H=H,
        W=W,
        HW=HW,
        inv_HW=inv_HW,
        stride_b=stride_b,
        stride_c=stride_c,
        stride_h=stride_h,
        stride_w=stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output

def replacement_func():
    return hardtanh_avg_pool2d_flatten