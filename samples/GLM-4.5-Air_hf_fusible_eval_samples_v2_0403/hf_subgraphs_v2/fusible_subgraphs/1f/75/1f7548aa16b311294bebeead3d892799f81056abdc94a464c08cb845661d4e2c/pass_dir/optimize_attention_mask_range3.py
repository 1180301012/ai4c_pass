import torch
import triton
import triton.language as tl

def pattern(mask_tensor):
    """Optimize attention mask processing for range_size=3 (matching TinyLlama model)"""
    # Convert mask to bool on cuda - EXACT match from original
    tmp_2 = mask_tensor.to(device='cuda', dtype=torch.bool)
    
    # Create arange with exact parameters from original
    tmp_3 = torch.arange(3, device='cuda')
    tmp_3 += 0
    tmp_4 = tmp_3
    
    # Apply indexing - EXACT match from original
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    
    return tmp_5, tmp_4, tmp_2

def replacement_args(mask_tensor):
    return (mask_tensor,)

@triton.jit
def optimized_attention_mask_kernel(
    mask_ptr,
    indexed_out_ptr,
    arange_out_ptr,
    mask_shape_0,
    mask_shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program indices
    row_idx = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 3  # Fixed range size
    
    # Generate arange values directly to output
    arange_vals = offsets.to(tl.int64)
    tl.store(arange_out_ptr + row_idx * 3 + offsets, arange_vals, mask=mask)
    
    # Load and convert mask values to bool
    mask_vals = tl.load(mask_ptr + row_idx * mask_shape_1 + offsets, mask=mask)
    indexed_vals = mask_vals.to(tl.int1)  # Convert to bool equivalent
    
    # Store indexed results
    tl.store(indexed_out_ptr + row_idx * 3 + offsets, indexed_vals, mask=mask)

@torch.fx.wrap
def optimized_attention_mask(mask_tensor):
    # Get shapes
    mask_shape_0, mask_shape_1 = mask_tensor.shape
    indexed_result = torch.empty((mask_shape_0, 3), dtype=torch.bool, device='cuda')
    arange_result = torch.empty((mask_shape_0, 3), dtype=torch.int64, device='cuda')
    
    BLOCK_SIZE = 1024
    num_rows = mask_shape_0
    num_cols = (3 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_attention_mask_kernel[(num_rows, num_cols)](
        mask_ptr=mask_tensor,
        indexed_out_ptr=indexed_result,
        arange_out_ptr=arange_result,
        mask_shape_0=mask_shape_0,
        mask_shape_1=mask_shape_1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return indexed_result, arange_result, mask_tensor

def replacement_func():
    return optimized_attention_mask