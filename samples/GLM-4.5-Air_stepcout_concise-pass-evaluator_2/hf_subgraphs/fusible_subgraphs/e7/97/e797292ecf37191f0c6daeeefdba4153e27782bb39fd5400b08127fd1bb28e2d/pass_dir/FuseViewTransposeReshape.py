import torch
import triton
import triton.language as tl
import math

@triton.jit
def triton_reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    value_dim,
    final_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < final_size
    
    # Reshape from [batch_size, value_dim] to [1, 1, batch_size * value_dim]
    # This is essentially a direct copy with different indexing
    # input: [batch_size, value_dim] -> output: [1, 1, batch_size * value_dim]
    
    # Calculate which element in output corresponds to this offset
    # The output is [1, 1, batch_size * value_dim], so we just need linear indexing
    input_offset = offsets
    
    # Load from input (treat as 1D tensor for simplicity)
    input_data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Store to output
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def optimized_reshape(bmm_output):
    """Optimized reshape that replaces view + transpose + reshape sequence"""
    # The original sequence is: [batch_size, value_dim] -> view(1, batch_size, 1, value_dim) 
    # -> transpose(1,2) -> reshape(1, 1, batch_size * value_dim)
    # This is equivalent to direct reshape to [1, 1, batch_size * value_dim]
    
    batch_size, value_dim = bmm_output.shape
    final_shape = [1, 1, batch_size * value_dim]
    final_size = batch_size * value_dim
    
    # For small tensors, just use torch reshape
    if final_size <= 4096:
        return bmm_output.reshape(final_shape)
    
    # For larger tensors, use optimized Triton kernel
    out = torch.empty(final_shape, dtype=bmm_output.dtype, device=bmm_output.device)
    
    BLOCK_SIZE = 1024
    num_programs = (final_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    triton_reshape_kernel[(num_programs,)](
        input_ptr=bmm_output,
        output_ptr=out,
        batch_size=batch_size,
        value_dim=value_dim,
        final_size=final_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(bmm_output):
    """Pattern: view + transpose + reshape sequence"""
    # Original sequence from model.py:
    # tmp_3 = tmp_2.view(1, batch_size, 1, value_dim)
    # tmp_4 = tmp_3.transpose(1, 2)
    # tmp_5 = tmp_4.reshape(1, 1, batch_size * value_dim)
    
    # The exact pattern to match (based on model.py):
    tmp_3 = bmm_output.view(1, bmm_output.shape[0], 1, bmm_output.shape[1])
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.reshape(1, 1, bmm_output.shape[0] * bmm_output.shape[1])
    return tmp_5

def replacement_args(bmm_output):
    return (bmm_output,)

def replacement_func():
    return optimized_reshape