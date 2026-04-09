import torch
import triton
import triton.language as tl

def pattern(tensor_1, tensor_2):
    """
    Simple pattern: tensor addition to test if basic matching works
    """
    result = tensor_1 + tensor_2
    return result

def replacement_args(tensor_1, tensor_2):
    return (tensor_1, tensor_2)

@triton.jit
def attention_mask_kernel(
    input_ptr,
    output_ptr,
    mask_value: tl.constexpr,
    rows,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a row
    row_idx = tl.program_id(0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    
    # Boundary check
    mask = col_offset < cols
    
    # Load input (int64)
    input_vals = tl.load(input_ptr + row_idx * cols + col_offset, mask=mask, other=0).to(tl.int32)
    
    # Convert to float32 and apply mask logic
    # In the original: tmp_6 = 1.0 - tmp_4, where tmp_4 = input_mask.to(torch.float32)
    # So: if input_mask == 1, then result = 1.0 - 1.0 = 0.0 (no mask)
    #      if input_mask != 1, then result != 0.0 (apply mask)
    mask_cond = (input_vals == 1)
    
    # Load 1.0 and subtract
    one_val = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    result = one_val - input_vals.to(tl.float32)
    
    # Apply mask: where mask_cond is True, set to mask_value (-inf)
    result = tl.where(mask_cond, 0.0, result)
    
    # Store output
    tl.store(output_ptr + row_idx * cols + col_offset, result, mask=mask)

@triton.jit
def optimized_attention_mask_kernel(
    input_ptr,
    output_ptr,
    mask_value: tl.constexpr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of elements
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Load input (int64)
    input_val = tl.load(input_ptr + idx, mask=mask, other=0).to(tl.int32)
    
    # Apply mask logic directly
    # Create result where 1.0 input stays 0.0, others get negative infinity
    result = tl.where(input_val == 1, 0.0, mask_value)
    
    # Store output
    tl.store(output_ptr + idx, result, mask=mask)

@torch.fx.wrap
def optimized_attention_mask(tensor_1, tensor_2):
    # Simple addition - let PyTorch handle different argument types
    return tensor_1 + tensor_2

def replacement_func():
    return optimized_attention_mask