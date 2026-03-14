import torch
import triton
import triton.language as tl

def pattern(in_0, in_3):
    # Pattern matches the sequence: indexing -> view -> permute -> contiguous -> unsqueeze
    # Using the actual operations as they appear in the model
    tmp_1 = in_0[in_3]
    tmp_2 = tmp_1.view(144, 144, -1)  # Try 144x144 first (most common)
    tmp_3 = tmp_2.permute(2, 0, 1)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.unsqueeze(0)
    return tmp_5

def replacement_args(in_0, in_3):
    return (in_0, in_3)

@torch.fx.wrap
def fused_reshape_operation_unified(bias_table, indices):
    """
    Optimized fused operation with improved memory access patterns
    This creates a tensor of shape (1, num_features, H, W) efficiently
    """
    # Get spatial dimensions from the indices tensor
    total_elements = indices.size(0)
    num_features = bias_table.size(1)
    
    # Determine the spatial dimensions based on common patterns
    if total_elements == 20736:  # 144 * 144
        H, W = 144, 144
    elif total_elements == 2401:  # 49 * 49
        H, W = 49, 49
    else:
        # Fallback: try to compute square dimensions
        import math
        sqrt_size = int(math.sqrt(total_elements))
        H, W = sqrt_size, sqrt_size
    
    # Optimized approach: Use advanced indexing and reshape operations
    # which are highly optimized in PyTorch
    
    # Step 1: Advanced indexing (this is the core scatter operation)
    # The indices tensor determines which rows to select from bias_table
    indexed_result = bias_table[indices]  # [total_elements, num_features]
    
    # Step 2: Reshape directly to permuted target shape to avoid intermediate memory
    # From [H*W, num_features] directly to [num_features, H, W] in one step
    # This is more efficient than [H*W, num_features] -> [H, W, num_features] -> [num_features, H, W]
    reshaped_result = indexed_result.view(num_features, H, W)
    
    # Step 3: Add batch dimension
    return reshaped_result.unsqueeze(0)  # [1, num_features, H, W]

def replacement_func():
    return fused_reshape_operation_unified