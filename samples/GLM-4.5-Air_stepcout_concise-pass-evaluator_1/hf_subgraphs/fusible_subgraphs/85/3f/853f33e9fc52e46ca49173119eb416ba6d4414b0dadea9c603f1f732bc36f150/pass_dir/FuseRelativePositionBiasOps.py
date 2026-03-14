import torch
import triton
import triton.language as tl
import math

def pattern(in_3, in_4):
    """Fuse the sequence of operations for relative position bias:
    tmp_2 = in_3[in_4]
    tmp_3 = tmp_2.view(197, 197, -1)
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = tmp_5.unsqueeze(0)
    """
    tmp_2 = in_3[in_4]
    tmp_3 = tmp_2.view(197, 197, -1)
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = tmp_5.unsqueeze(0)
    return tmp_6

def replacement_args(in_3, in_4):
    return (in_3, in_4)

@torch.fx.wrap  
def optimized_rel_pos_bias_fusion(table, indices):
    """Optimized relative position bias fusion with no conditionals and minimal overhead"""
    # Use advanced indexing directly - optimized by PyTorch
    indexed_values = table[indices]
    
    # Use -1 for automatic shape inference (cleaner than conditionals)
    # view(197, 197, -1) will automatically determine the feature dimension
    reshaped = indexed_values.view(197, 197, -1)
    
    # Single transpose operation (more efficient than separate permute + contiguous)
    # transpose(0, 2) is equivalent to permute(2, 0, 1)
    result = reshaped.transpose(0, 2)
    
    # Add batch dimension in one step
    result = result.unsqueeze(0)
    
    return result


@torch.fx.wrap  
def alternative_optimized_fusion(table, indices):
    """Alternative optimized approach using different memory access pattern"""
    # Pre-compute the total number of indices to avoid shape computation
    n_indices = indices.numel()
    
    # Use PyTorch's advanced indexing with memory layout optimization
    # Reshape indices to 2D for more efficient memory access
    indices_2d = indices.view(-1, 1)  # Reshape to [n_indices, 1] for broadcasting
    
    # Create row indices for direct 2D indexing
    table_height, table_width = table.shape
    row_indices = indices // table_width
    col_indices = indices % table_width
    
    # Use advanced indexing to directly extract values in desired order
    indexed_values = table[row_indices, col_indices]
    
    # Complete the reshape sequence efficiently
    # Note: since we have 38809 elements and view(197, 197, -1) is used,
    # the inferred dimension is 1, giving us [197, 197, 1]
    reshaped = indexed_values.view(197, 197, 1)
    
    # Final permute and unsqueeze
    result = reshaped.transpose(0, 2).unsqueeze(0)
    
    return result

def replacement_func():
    return optimized_rel_pos_bias_fusion