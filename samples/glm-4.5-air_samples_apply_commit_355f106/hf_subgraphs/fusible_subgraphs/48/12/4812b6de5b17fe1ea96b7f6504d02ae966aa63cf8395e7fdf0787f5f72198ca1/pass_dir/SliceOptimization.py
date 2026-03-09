import torch
import triton
import triton.language as tl

def pattern(input_tensor, start_idx):
    # The exact pattern that matches tensor slicing: slice(None, None, None), slice(start_idx, None, None), ...
    # This is equivalent to input_tensor[:, start_idx:, :, :]
    output = input_tensor[slice(None, None, None), slice(start_idx, None, None), slice(None, None, None), slice(None, None, None)]
    return output

def replacement_args(input_tensor, start_idx):
    return (input_tensor, start_idx)

# Optimized Triton kernel for tensor slicing
@triton.jit
def slice_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    original_N,
    original_C,
    original_H,
    original_W,
    start_idx,
    output_C,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate the output shape and compute input offset
    # The output tensor is [N, output_C, H, W] where output_C = original_C - start_idx
    total_per_channel_H_W = original_H * original_W
    total_per_slice = output_C * total_per_channel_H_W
    
    # Calculate which slice and which position within the slice we're handling
    slice_idx = offsets // total_per_slice
    remaining = offsets % total_per_slice
    
    channel_idx = start_idx + (remaining // total_per_channel_H_W)
    hw_idx = remaining % total_per_channel_H_W
    
    # Compute global input offset: [N, C, H, W] -> N*original_C*original_H*original_W + C*original_H*original_W + H*original_W + W
    input_offset = slice_idx * original_C * total_per_channel_H_W + channel_idx * total_per_channel_H_W + hw_idx
    
    # Load input data and store to output
    input_data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def triton_slice(input_tensor, start_idx):
    # Input tensor shape: [N, C, H, W]
    original_N, original_C, original_H, original_W = input_tensor.shape
    output_C = original_C - start_idx
    
    # Create output tensor
    output_shape = (original_N, output_C, original_H, original_W)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    n_elements = original_N * output_C * original_H * original_W
    
    # Calculate optimal block size and grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (num_programs,)
    
    # Launch the kernel
    slice_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        original_N=original_N,
        original_C=original_C,
        original_H=original_H,
        original_W=original_W,
        start_idx=start_idx,
        output_C=output_C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return triton_slice