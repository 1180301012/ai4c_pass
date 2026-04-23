import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(x):
    a = x.view(x.shape[0], 2, 20, x.shape[2], x.shape[3])
    b = torch.transpose(a, 1, 2)
    c = b.contiguous()
    d = c.view(x.shape[0], 40, x.shape[2], x.shape[3])
    return d

# Argument extraction function

def replacement_args(tmp_5):
    return (tmp_5,)

# Triton kernel for channel permutation
@triton.jit
def permute_channels_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    height,
    width,
    stride_batch,
    stride_c_in,
    stride_h,
    stride_w,
    stride_c_out,
    BLOCK_C: tl.constexpr
):
    
    # Calculate grid indices for the tensor
    batch_id = tl.program_id(0)
    h_id = tl.program_id(1)
    w_id = tl.program_id(2)
    c_block = tl.program_id(3)

    # Calculate starting channel for this block
    c_start = c_block * BLOCK_C

    # For each channel in the block
    for c in range(c_start, min(c_start + BLOCK_C, 40)):
        # Calculate the corresponding input channel for output channel c
        # Formula: input_channel = 2 * (c % 2) + (c // 2)
        input_c = 2 * (c % 2) + (c // 2)

        # Calculate input and output offsets
        input_offset = batch_id * stride_batch + input_c * stride_c_in + h_id * stride_h + w_id * stride_w
        output_offset = batch_id * stride_batch + c * stride_c_out + h_id * stride_h + w_id * stride_w

        # Load and store values
        in_val = tl.load(in_ptr + input_offset)
        tl.store(out_ptr + output_offset, in_val)

# Kernel wrapper
@torch.fx.wrap

def permute_channels_kernel_wrapper(tmp_5):
    # Get tensor shapes
    batch_size = tmp_5.shape[0]
    channels = tmp_5.shape[1]
    height = tmp_5.shape[2]
    width = tmp_5.shape[3]

    # Calculate strides for tensor
    stride_batch = tmp_5.stride(0)
    stride_c_in = tmp_5.stride(1)
    stride_h = tmp_5.stride(2)
    stride_w = tmp_5.stride(3)

    # Allocate output tensor
    out = torch.empty_like(tmp_5)

    # Calculate grid configuration
    num_batch = batch_size
    num_height = height
    num_width = width
    num_c_blocks = (40 + 16 - 1) // 16  # BLOCK_C = 16

    # Configure kernel launch
    grid = (num_batch, num_height, num_width, num_c_blocks)

    # Launch kernel
    permute_channels_kernel[grid](
        tmp_5,
        out,
        batch_size,
        height,
        width,
        stride_batch,
        stride_c_in,
        stride_h,
        stride_w,
        out.stride(1),
        BLOCK_C=16
    )

    return out

# Replacement function

def replacement_func():
    return permute_channels_kernel_wrapper