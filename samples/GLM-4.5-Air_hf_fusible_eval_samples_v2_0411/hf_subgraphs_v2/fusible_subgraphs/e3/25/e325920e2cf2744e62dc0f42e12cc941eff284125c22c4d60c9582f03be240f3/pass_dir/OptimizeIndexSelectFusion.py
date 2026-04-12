import torch
import triton
import triton.language as tl

def pattern(features, indices):
    # Try minimal pattern first - just index_select
    return features.index_select(-2, indices)

def replacement_args(features, indices):
    return (features, indices)

@triton.jit
def optimized_index_select_kernel(
    features_ptr,     # [num_nodes, feature_dim] tensor
    indices_ptr,      # [num_indices] tensor of indices
    output_ptr,       # [num_indices, feature_dim] output tensor
    num_nodes: tl.constexpr,
    num_indices: tl.constexpr,
    feature_dim: tl.constexpr,
):
    # Use 2D grid with better occupancy  
    pid = tl.program_id(0)
    feat_dim = tl.program_id(1)
    
    # Each program handles one feature element for better memory coalescing
    if feat_dim >= feature_dim:
        return
    
    # Process multiple indices per program for better occupancy
    num_total_elements = num_indices * feature_dim
    base_offset = pid
    step_size = tl.num_programs(0) * feature_dim
    
    # Process elements in chunks
    while base_offset < num_total_elements:
        idx_offset = base_offset // feature_dim  # which index this belongs to
        feat_offset = base_offset % feature_dim   # which feature in that index
        
        # Only process if within bounds
        if idx_offset < num_indices:
            # Load the index
            index = tl.load(indices_ptr + idx_offset)
            
            # Check bounds
            mask = (index < num_nodes) & (index >= 0)
            
            # Load single feature element with bounds checking using masked load
            feat_ptr = features_ptr + index * feature_dim + feat_offset
            output_ptr_pos = output_ptr + idx_offset * feature_dim + feat_offset
            
            # Feature offset bounds check
            feature_offset_valid = (index * feature_dim + feat_offset) < (num_nodes * feature_dim)
            final_mask = mask & feature_offset_valid
            
            # Use masked load with other=0.0 to handle both types correctly
            feature = tl.load(feat_ptr, mask=final_mask, other=0.0)
            tl.store(output_ptr_pos, feature, mask=final_mask)
        
        # Move to next chunk
        base_offset += step_size

@torch.fx.wrap
def optimized_index_select(features, indices):
    num_nodes = features.shape[0]
    num_indices = indices.shape[0]
    feature_dim = features.shape[1]
    
    # Handle both bfloat16 and float16 dtypes
    if features.dtype == torch.bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    # Allocate output
    output = torch.empty(num_indices, feature_dim, dtype=dtype, device=features.device)
    
    # Use 2D grid for better memory coalescing
    # First dimension: programs handling total elements
    # Second dimension: programs handling different features
    grid_x = (num_indices * feature_dim + 255) // 256  # Launch multiple of 256 for occupancy
    grid_y = feature_dim
    grid = (grid_x, grid_y)
    
    # Launch kernel
    optimized_index_select_kernel[grid](
        features,
        indices,
        output,
        num_nodes,
        num_indices,
        feature_dim,
    )
    
    return output

def replacement_func():
    return optimized_index_select