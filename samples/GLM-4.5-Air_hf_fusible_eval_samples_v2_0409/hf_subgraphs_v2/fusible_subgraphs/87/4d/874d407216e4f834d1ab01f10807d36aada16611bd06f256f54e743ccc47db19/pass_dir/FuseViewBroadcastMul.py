import torch
import triton
import triton.language as tl

def pattern(weights, features):
    """
    Pattern: view(-1, 1) followed by multiplication with broadcasting
    This matches where we reshape weights to [N, 1] and multiply with features [N, D] 
    to get broadcasted result [N, D]
    """
    tmp = weights.view(-1, 1)
    result = tmp * features
    return result

def replacement_args(weights, features):
    return (weights, features)

@triton.jit
def optimized_broadcast_mul_kernel(
    weights_ptr,
    features_ptr,
    out_ptr,
    n_elements,
    feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate row and column indices
    col_offsets = tl.arange(0, feature_dim)
    row_idx_offsets = offsets // feature_dim
    col_idx_offsets = offsets % feature_dim
    
    # Broadcast: each weight multiplies all features in its row
    weights_broadcasted = tl.load(weights_ptr + row_idx_offsets, mask=row_idx_offsets < n_elements // feature_dim)
    features_loaded = tl.load(features_ptr + offsets, mask=mask)
    
    # Apply broadcasting multiplication
    out = weights_broadcasted * features_loaded
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_broadcast_mul(weights, features):
    """
    Optimized broadcast multiplication that handles the view(-1, 1) * features pattern
    directly without creating intermediate tensor
    """
    n_elements = weights.numel() * features.shape[1]  # N * D
    feature_dim = features.shape[1]
    
    # Tune block size for optimal performance
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(features)
    
    # Launch kernel
    optimized_broadcast_mul_kernel[(num_programs,)](
        weights_ptr=weights,
        features_ptr=features,
        out_ptr=out,
        n_elements=n_elements,
        feature_dim=feature_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    def optimized_func(weights, features):
        return optimized_broadcast_mul(weights, features)
    return optimized_func