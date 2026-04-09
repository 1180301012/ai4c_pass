import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern matching for tensor slicing operation - extracting odd indices"""
    # This matches the RoPE slicing pattern: x[(Ellipsis, slice(1, None, 2))]
    return x[(Ellipsis, slice(1, None, 2))]

def replacement_args(x):
    return (x,)

@triton.jit
def slice_odd_kernel(
    input_ptr,
    output_ptr,
    input_size,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # For odd indexing, output element i corresponds to input element 2*i + 1
    input_offsets = offsets * 2 + 1
    input_mask = input_offsets < input_size
    
    # Load from input at odd indices, store to output
    input_data = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def optimized_slice_odd(x):
    """Optimized slicing to extract odd indices using Triton"""
    # For odd indices: result[i] = x[2*i + 1]
    # Output size is roughly half of input size (rounding up)
    input_size = x.numel()
    output_size = (input_size + 1) // 2
    
    output = torch.empty(output_size, dtype=x.dtype, device=x.device)
    
    # Block size and grid configuration
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    slice_odd_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        input_size=input_size,
        output_size=output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_slice_odd