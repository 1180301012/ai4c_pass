import torch
import triton
import triton.language as tl

def pattern(inv_freq, arange_tensor):
    """
    Pattern matching for optimized outer product with self-concatenation.
    Matches the sequence:
    - tmp_3 = torch.outer(tmp_2, tmp_0)
    - tmp_4 = torch.cat((tmp_3, tmp_3), dim=-1)
    
    Returns the concatenated result which is used in subsequent operations
    """
    # Outer product computation
    outer_product = torch.outer(arange_tensor, inv_freq)
    
    # Self-concatenation along last dimension
    concatenated_result = torch.cat((outer_product, outer_product), dim=-1)
    
    return concatenated_result

def replacement_args(inv_freq, arange_tensor):
    """Extract arguments for the optimized kernel"""
    return inv_freq, arange_tensor

@triton.jit
def optimized_outer_concat_kernel(
    inv_freq_ptr,
    arange_ptr,
    output_ptr,
    inv_freq_size,
    arange_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that computes outer product and self-concatenation in a single operation.
    This avoids creating the intermediate outer product tensor and immediately doubles the last dimension.
    """
    # Get program ID for 2D grid
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < arange_size
    
    # Load arange tensor data
    arange_data = tl.load(arange_ptr + offsets, mask=mask, other=0.0)
    
    # Compute indices for the full output (concatenated dimension)
    col_in_output = tl.arange(0, inv_freq_size * 2)
    inv_freq_idx = col_in_output % inv_freq_size
    
    # Load inv_freq data for all columns (reusing inv_freq data for concatenated parts)
    inv_freq_data = tl.load(inv_freq_ptr + inv_freq_idx, other=0.0)
    
    # Compute outer product result for all columns
    result = arange_data[:, None] * inv_freq_data[None, :]
    
    # Store results
    tl.store(output_ptr + offsets[:, None] + inv_freq_idx, result, mask=mask[:, None])

@torch.fx.wrap
def optimized_outer_concat_wrapper(inv_freq, arange_tensor):
    """Wrapper function to launch the optimized outer product + concatenation kernel"""
    # Get tensor properties
    inv_freq_size = inv_freq.shape[0]
    arange_size = arange_tensor.shape[0]
    
    # Create output tensor (concatenated dimensions: [arange_size, inv_freq_size * 2])
    output_size = arange_size * inv_freq_size * 2
    output = torch.empty((arange_size, inv_freq_size * 2), dtype=inv_freq.dtype, device=inv_freq.device)
    
    # Set block size and launch grid
    BLOCK_SIZE = 1024
    num_programs = (arange_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    optimized_outer_concat_kernel[(num_programs,)](
        inv_freq_ptr=inv_freq,
        arange_ptr=arange_tensor,
        output_ptr=output,
        inv_freq_size=inv_freq_size,
        arange_size=arange_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized outer product + concatenation function"""
    return optimized_outer_concat_wrapper