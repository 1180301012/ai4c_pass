import torch
import triton
import triton.language as tl
from typing import Union, Tuple


def pattern(x):
    # This pattern matches tmp_0[:, None, None, :] operation
    # which is adding singleton dimensions at positions 1 and 2
    result = x[slice(None, None, None), None, None, slice(None, None, None)]
    return result


def replacement_args(x):
    return (x,)


@triton.jit
def optimized_unsqueeze_kernel(
    x_ptr,
    out_ptr,
    original_batch_size,
    original_seq_len,
    output_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global memory offset
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Boundary check
    mask = idx < original_batch_size * original_seq_len
    
    # Load input data (flattened view of original tensor)
    x_val = tl.load(x_ptr + idx, mask=mask, other=0)
    
    # Calculate output offset (same for all added singleton dims)
    out_offset = idx  # Since singleton dims don't change data location
    
    # Store to output (same shape data, just with singleton dims)
    tl.store(out_ptr + out_offset, x_val, mask=mask)


@torch.fx.wrap  
def optimized_unsqueeze(x):
    """
    Optimized version of x[:, None, None, :]
    This operation expands dimension 1 and 2 with size 1
    """
    original_shape = x.shape
    batch_size = original_shape[0] 
    seq_len = original_shape[1] if len(original_shape) > 1 else 1
    
    # New shape will be [batch_size, 1, 1, seq_len] for 2D input
    if len(original_shape) == 2:
        new_shape = (batch_size, 1, 1, seq_len)
    else:
        # For other shapes, just return the unsqueezed version
        return x[slice(None, None, None), None, None, slice(None, None, None)]
    
    # Create output tensor
    out = torch.empty(new_shape, dtype=x.dtype, device=x.device)
    
    # Calculate kernel parameters
    total_elements = batch_size * seq_len
    grid = ( (total_elements + 255) // 256, )  # Using 256 block size
    
    # Launch kernel only if it's beneficial
    if total_elements > 1024:  # Only use kernel for reasonable sizes
        optimized_unsqueeze_kernel[grid](
            x_ptr=x,
            out_ptr=out,
            original_batch_size=batch_size,
            original_seq_len=seq_len,
            output_seq_len=seq_len,
            BLOCK_SIZE=256,
        )
    else:
        # For small tensors, use standard operation
        out[:] = x[slice(None, None, None), None, None, slice(None, None, None)][:]
    
    return out


def replacement_func():
    return optimized_unsqueeze