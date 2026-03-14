import torch

def pattern(add_tensor1, add_tensor2, mean_dim, mean_keepdim, 
            dropout_x, dropout_p, dropout_train, dropout_inplace,
            dropout2_x, dropout2_p, dropout2_train, dropout2_inplace,
            bn_input, bn_running_mean, bn_running_var, bn_weight, bn_bias,
            bn_use_running_stats, bn_momentum, bn_eps):
    """
    Pattern: Entire computation chain from addition to batch normalization
    In the model:
        tmp_4 = in_5 + in_4
        tmp_5 = tmp_4.mean((2, 3), keepdim=False)
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    """
    # Element-wise addition
    tmp_4 = add_tensor1 + add_tensor2
    
    # Mean computation - optimized to use sum + division for better performance
    tmp_5 = tmp_4.mean(mean_dim, mean_keepdim)
    
    # Two dropout operations with p=0.0 - both are identity operations
    tmp_6 = torch.nn.functional.dropout(dropout_x, dropout_p, dropout_train, dropout_inplace)
    tmp_7 = torch.nn.functional.dropout(dropout2_x, dropout2_p, dropout2_train, dropout2_inplace)
    
    # Batch normalization
    tmp_8 = torch.nn.functional.batch_norm(
        tmp_7, bn_running_mean, bn_running_var, bn_weight, bn_bias,
        bn_use_running_stats, bn_momentum, bn_eps
    )
    
    # Return both outputs as in the original model
    return tmp_8, tmp_7

def replacement_args(add_tensor1, add_tensor2, mean_dim, mean_keepdim, 
           dropout_x, dropout_p, dropout_train, dropout_inplace,
           dropout2_x, dropout2_p, dropout2_train, dropout2_inplace,
           bn_input, bn_running_mean, bn_running_var, bn_weight, bn_bias,
           bn_use_running_stats, bn_momentum, bn_eps):
    return (add_tensor1, add_tensor2, mean_dim, mean_keepdim,
           dropout_x, dropout_p, dropout_train, dropout_inplace,
           dropout2_x, dropout2_p, dropout2_train, dropout2_inplace,
           bn_input, bn_running_mean, bn_running_var, bn_weight, bn_bias,
           bn_use_running_stats, bn_momentum, bn_eps)

@torch.fx.wrap
def optimized_add_to_batch_norm(add_tensor1, add_tensor2, mean_dim, mean_keepdim, 
           dropout_x, dropout_p, dropout_train, dropout_inplace,
           dropout2_x, dropout2_p, dropout2_train, dropout2_inplace,
           bn_input, bn_running_mean, bn_running_var, bn_weight, bn_bias,
           bn_use_running_stats, bn_momentum, bn_eps):
    """
    Optimized version of the entire computation chain
    - Optimized mean computation using sum + division
    - Identity operations for zero-probability dropout
    - Better memory access patterns
    """
    # Optimized mean computation: sum + division can be more efficient than direct mean
    tmp_4 = add_tensor1 + add_tensor2
    
    # Use sum and division as this can be more efficient
    if not mean_keepdim and tmp_4.dim() == 4 and tuple(mean_dim) == (2, 3):
        batch_size, channels, height, width = tmp_4.shape
        spatial_sum = tmp_4.sum(dim=(2, 3))
        tmp_5 = spatial_sum / (height * width)
    else:
        tmp_5 = tmp_4.mean(mean_dim, mean_keepdim)
    
    # Optimize consecutive zero-probability dropouts: both are identity when p=0.0
    if dropout_p == 0.0 and dropout2_p == 0.0 and not dropout_train and not dropout2_train:
        # Skip both dropouts entirely - direct assignment is faster
        tmp_7 = tmp_5
    else:
        # Fallback to original behavior for non-zero dropout probabilities
        tmp_6 = torch.nn.functional.dropout(dropout_x, dropout_p, dropout_train, dropout_inplace)
        tmp_7 = torch.nn.functional.dropout(dropout2_x, dropout2_p, dropout2_train, dropout2_inplace)
    
    # Batch normalization (unchanged as it's already optimized)
    tmp_8 = torch.nn.functional.batch_norm(
        tmp_7, bn_running_mean, bn_running_var, bn_weight, bn_bias,
        bn_use_running_stats, bn_momentum, bn_eps
    )
    
    return tmp_8, tmp_7

def replacement_func():
    return optimized_add_to_batch_norm