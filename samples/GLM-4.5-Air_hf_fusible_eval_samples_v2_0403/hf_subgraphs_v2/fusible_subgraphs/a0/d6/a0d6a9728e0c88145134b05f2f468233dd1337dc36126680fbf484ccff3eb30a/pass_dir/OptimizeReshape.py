import torch

# Pattern matching for view-transpose-reshape sequence
def pattern(bmm_1):
    # Original sequence: view -> transpose -> reshape
    tmp_4 = bmm_1.view(1, -1, 1, -1)  # Flexible to handle different K, M dimensions
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, -1)  # Final reshape to flattened dimension
    return tmp_4, tmp_5, tmp_6

# Arguments needed for replacement
def replacement_args(bmm_1):
    return (bmm_1,)

# Optimized function that eliminates intermediate steps
def optimized_reshape(bmm_1):
    # The sequence view(1, K, 1, M) -> transpose(1,2) -> reshape(1, 1, K*M)
    # is equivalent to directly reshaping to (1, 1, -1)
    # since K*M equals the total flattened dimension from bmm_1's last two dimensions
    
    # Original bmm_1 shape is [batch, seq_len, dim]
    # We need to flatten the last two dimensions and reshape to [1, 1, batch*seq_len*dim]
    # However, looking at the specific pattern, it flattens only the last two dimensions
    # and then reshapes to [1, 1, -1]
    
    # Calculate the flattened dimension
    _, _, dim = bmm_1.shape
    seq_len = bmm_1.shape[1]  # This will be the "K" dimension
    _, _, second_dim = bmm_1.shape  # This will be the "M" dimension
    
    # Direct reshape to the target format
    # Note: The view pattern in the original code flattens the last two dimensions
    # and then reshapes to [1, 1, K*M] where K*M = seq_len * second_dim
    original_final_size = seq_len * second_dim
    
    # Directly reshape to the final output
    result = bmm_1.reshape(1, 1, -1)  # This handles all cases correctly
    
    return result

# Wrapper function for the replacement
@torch.fx.wrap
def reshape_optimization_wrapper(bmm_1):
    return optimized_reshape(bmm_1)

# Replacement function (returns function reference, not a call)
def replacement_func():
    return reshape_optimization_wrapper