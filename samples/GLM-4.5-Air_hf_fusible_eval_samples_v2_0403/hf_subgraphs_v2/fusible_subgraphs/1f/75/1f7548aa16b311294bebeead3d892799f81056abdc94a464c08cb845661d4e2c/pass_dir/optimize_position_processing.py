import torch
import triton
import triton.language as tl
from torch import device

def position_shape_inference(tensor):
    """Determine the optimal shape for position processing"""
    if tensor.dim() == 1:
        return (tensor.numel(), 1)
    else:
        return (tensor.numel(), 1)

def pattern(cache_position_tensor, range_size):
    """Optimize position/cache processing with fused arange and comparison"""
    # Create first arange on cuda
    tmp_3 = torch.arange(range_size, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    
    # Handle arange - this is just a view of the original
    tmp_6 = torch.arange(range_size, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    
    # View cache_position as column vector and compare with arange
    tmp_8 = cache_position_tensor.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    
    # Format: (first_arange, comparison_result, view_result)
    return tmp_4, tmp_9, tmp_8

def replacement_args(cache_position_tensor, range_size):
    return (cache_position_tensor, range_size)

@triton.jit
def optimized_position_kernel(
    cache_pos_ptr,
    range_size,
    arange_out_ptr,
    comparison_out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate thread indices
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < range_size
    
    # Generate arange values directly
    arange_vals = row_idx.to(tl.int64)
    tl.store(arange_out_ptr + row_idx, arange_vals, mask=mask)
    
    # Load cache position values (broadcasted comparison)
    cache_pos_val = tl.load(cache_pos_ptr, mask=tl.program_id(0) == 0)  # Load first element only
    cache_pos_broadcasted = cache_pos_val  # This will be broadcasted in comparison
    
    # Perform comparison (arange <= cache_position)
    comparison_vals = arange_vals <= cache_pos_broadcasted
    tl.store(comparison_out_ptr + row_idx, comparison_vals, mask=mask)

@torch.fx.wrap
def optimized_position_processing(cache_position_tensor, range_size):
    # Get optimal shape
    optimal_shape = position_shape_inference(cache_position_tensor)
    
    # Prepare output tensors
    arange_result = torch.empty(range_size, dtype=torch.int64, device='cuda')
    comparison_result = torch.empty(range_size, dtype=torch.bool, device='cuda')
    
    BLOCK_SIZE = 1024
    num_programs = (range_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernels
    optimized_position_kernel[(num_programs,)](
        cache_pos_ptr=cache_position_tensor,
        range_size=range_size,
        arange_out_ptr=arange_result,
        comparison_out_ptr=comparison_result,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply view expansion to match original pattern
    view_result = cache_position_tensor.view(-1, 1)
    
    return arange_result, comparison_result, view_result

def replacement_func():
    return optimized_position_processing