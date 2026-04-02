import torch
import triton
import triton.language as tl

# Pattern matching for final masking operation (lines 28-29)
def pattern(triangular_mask, bool_mask):
    tmp_19 = triangular_mask.masked_fill(bool_mask, -3.4028234663852886e+38)
    return tmp_19

@triton.jit
def apply_final_mask_kernel(
    triangular_ptr,
    bool_ptr,
    output_ptr,
    mask_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to avoid out-of-bounds access
    mask = offsets < triangular_ptr.size(0)
    
    # Load triangular mask
    triangular_val = tl.load(triangular_ptr + offsets, mask=mask, other=0.0)
    
    # Load boolean mask (stored as float32, need to convert)
    bool_val = tl.load(bool_ptr + offsets, mask=mask, other=0.0)
    bool_mask = (bool_val != 0.0)
    
    # Apply masking: where bool is True, set to mask_value
    result = tl.where(bool_mask, mask_value, triangular_val)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_final_masking(triangular_mask, bool_mask):
    # Ensure both tensors have the same shape
    if triangular_mask.shape != bool_mask.shape:
        # This shouldn't happen in the matched pattern, but just in case
        bool_mask = bool_mask.expand_as(triangular_mask)
    
    num_elements = triangular_mask.numel()
    
    # Create output tensor
    output = torch.empty_like(triangular_mask)
    
    # Set up grid
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    apply_final_mask_kernel[(num_programs,)](
        triangular_ptr=triangular_mask,
        bool_ptr=bool_mask,
        output_ptr=output,
        mask_value=-3.4028234663852886e+38,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_args(triangular_mask, bool_mask):
    return (triangular_mask, bool_mask)

def replacement_func():
    return optimized_final_masking