import torch
import triton
import triton.language as tl

# Pattern matching function - matches two consecutive type conversions to float16
def pattern(cos_result, sin_result):
    # Now that multiplications by 1.0 are eliminated, this directly matches
    # cos and sin results being converted to float16
    return cos_result.to(dtype=torch.float16), sin_result.to(dtype=torch.float16)

# Argument extraction function
def replacement_args(cos_result, sin_result):
    return (cos_result, sin_result)

# Optimized kernel for fused type conversion
@triton.jit
def fused_conversion_kernel(
    cos_ptr,
    sin_ptr, 
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load both cos and sin data
    cos_val = tl.load(cos_ptr + offsets, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptr + offsets, mask=mask, other=0.0)
    
    # Triton automatically handles float32 to float16 conversion when storing to float16 tensor
    # So just store directly - the dtype conversion happens implicitly
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)

@torch.fx.wrap
def fused_type_conversion(cos_result, sin_result):
    # Get shape information
    *batch_dims, seq_len, hidden_dim = cos_result.shape
    total_elements = batch_dims[0] * seq_len * hidden_dim
    
    # Create output tensors as float16
    cos_out = torch.empty_like(cos_result, dtype=torch.float16)
    sin_out = torch.empty_like(sin_result, dtype=torch.float16)
    
    # Set block size and launch grid
    BLOCK_SIZE = 1024  # Larger block size for memory bandwidth bound operations
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conversion_kernel[(num_programs,)](
        cos_ptr=cos_result,
        sin_ptr=sin_result,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out, sin_out

def replacement_func():
    return fused_type_conversion