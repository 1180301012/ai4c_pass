import torch
import triton
import triton.language as tl

def pattern(in_1):
    """Pattern for fused sin/cos operations with concatenation and type conversions"""
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return (tmp_6, tmp_7)

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def fused_trigonometric_kernel(
    input_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for computing both sin and cos with concatenated input and type conversion"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Concatenate with itself by replicating the values (in-place operation)
    # For even indices, we use the original values, for odd indices, we replicate
    concat_indices = offsets * 2
    concat_mask = concat_indices < (n_elements * 2)
    x_concat_even = tl.load(input_ptr + concat_indices, mask=concat_mask, other=0.0)
    x_concat_odd = x_concat_even  # Same values for both positions
    
    # Alternate between even and odd indices for the concatenated array
    full_offsets = concat_indices
    cos_values = tl.cos(x_concat_even)
    sin_values = tl.sin(x_concat_even)
    
    # Apply identity multiplication and convert to bfloat16
    cos_out = cos_values.to(tl.bfloat16)
    sin_out = sin_values.to(tl.bfloat16)
    
    # Store results
    tl.store(cos_out_ptr + full_offsets, cos_out, mask=concat_mask)
    tl.store(sin_out_ptr + full_offsets, sin_out, mask=concat_mask)

@torch.fx.wrap
def fused_trigonometric_ops(in_1):
    """Wrapper function for fused trigonometric operations"""
    # Get input shape and calculate total elements after concatenation
    original_shape = in_1.shape
    original_size = in_1.numel()
    concatenated_size = original_size * 2
    
    # Create output tensors
    cos_out = torch.empty(original_shape, dtype=torch.bfloat16, device=in_1.device)
    sin_out = torch.empty(original_shape, dtype=torch.bfloat16, device=in_1.device)
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (concatenated_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_trigonometric_kernel[(num_programs,)](
        input_ptr=in_1,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=original_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (cos_out, sin_out)

def replacement_func():
    return fused_trigonometric_ops