import torch
import triton
import triton.language as tl

def pattern(in_12):
    """
    Pattern to match attention mask processing operations.
    This matches: 
    tmp_12 = in_12.to(dtype=torch.float32)
    tmp_13 = 1.0 - tmp_12
    tmp_14 = tmp_13 * -3.4028234663852886e+38
    """
    tmp_12 = in_12.to(dtype=torch.float32)
    tmp_13 = 1.0 - tmp_12
    tmp_14 = tmp_13 * -3.4028234663852886e+38
    
    return tmp_14

def replacement_args(in_12):
    return (in_12,)

@triton.jit
def attention_mask_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    mask_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized attention mask processing kernel"""
    idx = tl.program_id(0)
    offset = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load input as float32 for the computation
    input_val = tl.load(input_ptr + offset, mask=mask, other=0).to(tl.float32)
    
    # Apply mask transformation: (1.0 - input) * max_negative
    # This is typically used for attention masks where:
    # - input=1 means keep (not masked) -> 1.0-1=0 -> 0*max_negative=0
    # - input=0 means don't attend (masked) -> 1.0-0=1 -> 1*max_negative=max_negative
    output_val = (1.0 - input_val) * mask_value
    tl.store(output_ptr + offset, output_val, mask=mask)

@torch.fx.wrap
def optimized_attention_mask(input_tensor, mask_value=-3.4028234663852886e+38):
    """Wrapper for optimized attention mask processing"""
    # Convert to float32 first if needed
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(torch.float32)
    
    output = torch.empty_like(input_tensor)
    
    # Calculate grid size
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    attention_mask_kernel[(grid_size,)](
        input_tensor,
        output,
        n_elements,
        mask_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_attention_mask