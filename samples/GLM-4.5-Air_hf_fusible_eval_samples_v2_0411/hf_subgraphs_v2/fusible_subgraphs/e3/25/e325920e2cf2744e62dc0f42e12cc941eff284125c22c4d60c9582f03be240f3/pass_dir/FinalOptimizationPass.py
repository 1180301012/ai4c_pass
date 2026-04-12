import torch
import triton
import triton.language as tl

def pattern(features, indices):
    # Simple but effective pattern for index_select
    return features.index_select(-2, indices)

def replacement_args(features, indices):
    return (features, indices)

@triton.jit
def optimized_index_select_kernel(
    features_ptr,
    indices_ptr, 
    output_ptr,
    num_nodes: tl.constexpr,
    num_indices: tl.constexpr,
    feature_dim: tl.constexpr,
):
    # Simple 2D grid with minimal overhead
    pid = tl.program_id(0)
    feat_dim = tl.program_id(1)
    
    # Exit if beyond feature dimension
    if feat_dim >= feature_dim:
        return
    
    # Process one element per program for maximum simplicity
    if pid < num_indices:
        # Load the index
        index = tl.load(indices_ptr + pid)
        
        # Check bounds
        mask = (index >= 0) & (index < num_nodes)
        
        if mask:
            # Calculate memory offsets
            input_offset = index * feature_dim + feat_dim
            output_offset = pid * feature_dim + feat_dim
            
            # Bounds check input offset
            input_valid = input_offset < (num_nodes * feature_dim)
            final_mask = mask & input_valid
            
            # Use masked load to handle both dtypes correctly
            feature = tl.load(features_ptr + input_offset, mask=final_mask, other=0.0)
            tl.store(output_ptr + output_offset, feature, mask=final_mask)

@torch.fx.wrap
def optimized_index_select(features, indices):
    num_nodes = features.shape[0]
    num_indices = indices.shape[0] 
    feature_dim = features.shape[1]
    
    # Handle both dtypes
    dtype = features.dtype
    
    # Allocate output
    output = torch.empty(num_indices, feature_dim, dtype=dtype, device=features.device)
    
    # Simple 2D grid: one program per (index, feature) pair
    grid_x = num_indices
    grid_y = feature_dim
    grid = (grid_x, grid_y)
    
    # Launch kernel
    optimized_index_select_kernel[grid](
        features, indices, output,
        num_nodes, num_indices, feature_dim,
    )
    
    return output

def replacement_func():
    return optimized_index_select