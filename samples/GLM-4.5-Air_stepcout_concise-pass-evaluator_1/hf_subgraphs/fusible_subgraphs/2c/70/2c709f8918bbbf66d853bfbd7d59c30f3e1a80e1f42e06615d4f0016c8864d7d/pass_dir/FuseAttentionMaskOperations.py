import torch
import triton
import triton.language as tl

def pattern(y):
    # Start with a simpler pattern - just match the masking operations:
    # tmp_1 = y.to(dtype=torch.float32)
    # tmp_2 = 1.0 - tmp_1  
    # tmp_3 = tmp_2 * -3.4028234663852886e+38
    tmp_1 = y.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    return tmp_3

def replacement_args(y):
    return (y,)

@triton.jit
def fused_attention_mask_kernel(
    mask_ptr,           # Input mask tensor
    output_mask_ptr,    # Output fused mask result  
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the mask tensor with better memory access
    mask_val = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations vectorized for better performance
    # Convert to float32, subtract from 1.0, multiply by large negative constant
    fused_result = (1.0 - tl.cast(mask_val, tl.float32)) * -3.4028234663852886e+38
    
    # Store the result
    tl.store(output_mask_ptr + offsets, fused_result, mask=mask)

@torch.fx.wrap
def fused_attention_mask(y):
    n_elements = y.numel()
    
    # Create output tensor
    output_mask = torch.empty_like(y, dtype=torch.float32)
    
    # Use optimized block size based on tensor size
    if n_elements < 1024:
        BLOCK_SIZE = 256
    elif n_elements < 8192:
        BLOCK_SIZE = 512  
    else:
        BLOCK_SIZE = 1024
    
    # Calculate grid size
    if n_elements == 0:
        return output_mask
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the fused kernel with optimized block size
    fused_attention_mask_kernel[grid](
        y,
        output_mask,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_mask

def replacement_func():
    return fused_attention_mask