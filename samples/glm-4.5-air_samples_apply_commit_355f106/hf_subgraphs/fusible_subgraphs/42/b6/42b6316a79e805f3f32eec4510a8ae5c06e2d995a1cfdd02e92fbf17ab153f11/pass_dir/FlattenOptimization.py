import torch
import triton
import triton.language as tl

def pattern(input):
    flattened_out = input.flatten(1, -1)
    return flattened_out

def replacement_args(input):
    return (input,)

@triton.jit
def flatten_kernel(
    input_ptr, output_ptr,
    n_batch, n_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block within one batch element
    batch_id = tl.program_id(0)
    pid = tl.program_id(1)  # Program ID for parallel processing within batch
    
    # Calculate start position for this batch element
    input_base = input_ptr + batch_id * n_channels * height * width
    output_base = output_ptr + batch_id * n_channels * height * width
    
    # Calculate block start position
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_channels * height * width
    
    # For flatten, we just need to copy data from input to output
    # since flatten is just a view operation, but we want to optimize the memory access pattern
    tl.store(output_ptr + output_base + offsets, 
             tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0), 
             mask=mask)

@torch.fx.wrap
def optimized_flatten(input):
    n_batch, n_channels, height, width = input.shape
    
    # Total elements per batch
    elements_per_batch = n_channels * height * width
    
    # Configure block size for optimal performance
    BLOCK_SIZE = 1024
    
    # Calculate number of programs per batch
    programs_per_batch = (elements_per_batch + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor ( flattened: n_batch, n_channels * height * width )
    output_shape = (n_batch, n_channels * height * width)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Launch kernel - grid is (batch, programs_per_batch)
    grid = (n_batch, programs_per_batch)
    
    flatten_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        n_batch=n_batch,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_flatten