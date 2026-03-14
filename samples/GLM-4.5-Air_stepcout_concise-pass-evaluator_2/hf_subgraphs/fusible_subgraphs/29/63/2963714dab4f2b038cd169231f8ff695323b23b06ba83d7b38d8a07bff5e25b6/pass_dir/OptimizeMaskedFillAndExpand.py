import torch
import triton
import triton.language as tl

def pattern(tmp_2, tmp_3):
    """Optimizes the sequence: masked_fill -> unsqueeze -> expand"""
    tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    return tmp_4, tmp_6

def replacement_args(tmp_2, tmp_3):
    return (tmp_2, tmp_3)

@triton.jit
def optimized_masked_expand_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    batch_size: tl.constexpr,
):
    """Optimized kernel that combines masked fill with dimension expansion"""
    pid = tl.program_id(0)
    block_start = pid * 1024
    offsets = block_start + tl.arange(0, 1024)
    mask = offsets < n_elements
    
    # Load input and mask data
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=0)
    
    # Apply masked fill: if mask is True, fill with 1, else keep input
    output_vals = tl.where(mask_vals, 1, input_vals)
    
    # Store results for all expanded dimensions
    for b in range(batch_size):
        expanded_offset = b * n_elements + offsets
        tl.store(output_ptr + expanded_offset, output_vals, mask=mask)

@torch.fx.wrap
def optimized_masked_expand(tmp_2, tmp_3):
    n_elements = tmp_2.numel()
    batch_size = 3  # From expand(3, -1, -1)
    
    # Output should have shape [batch_size, *original_shape]
    original_shape = tmp_2.shape
    output_shape = (batch_size,) + original_shape
    output = torch.empty(output_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Launch kernel
    n_programs = (n_elements + 1023) // 1024
    optimized_masked_expand_kernel[(n_programs,)](
        input_ptr=tmp_2,
        mask_ptr=tmp_3,
        output_ptr=output,
        n_elements=n_elements,
        batch_size=batch_size,
    )
    
    return tmp_2, output  # Return original masked result and expanded version

def replacement_func():
    return optimized_masked_expand