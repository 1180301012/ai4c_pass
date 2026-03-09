import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    # Match the computation pattern:
    # Addition -> Split -> Extract first -> Transpose
    tmp_0 = in_1 + in_0
    tmp_1 = torch.functional.split(tmp_0, [1, -1], 1)
    tmp_0 = None  # Exclude cleanup operations
    tmp_2 = tmp_1[0]  # First part (unchanged)
    tmp_3 = tmp_1[1]  # Second part to be transposed
    tmp_1 = None  # Exclude cleanup operations
    tmp_4 = tmp_3.permute(0, 2, 1)  # Transpose
    tmp_3 = None  # Exclude cleanup operations
    return (tmp_2, tmp_4)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Wrapper function for fused operations
@torch.fx.wrap
def fused_split_transpose_view(in_0, in_1):
    # Optimized kernel without autotuning
    @triton.jit
    def fused_split_transpose_view_kernel(
        # Input tensors
        in_ptr,
        out_first_ptr,
        out_second_ptr,
        # Shape parameters
        batch_size,
        total_dim,
        feature_dim,
        spatial_dim,
        # Constants
        SPATIAL_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program handles a block of elements
        pid = tl.program_id(0)
        block_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total_elements = batch_size * total_dim * feature_dim
        
        # Create grid indices
        batch_idx = block_offset // (total_dim * feature_dim)
        batch_remainder = block_offset % (total_dim * feature_dim)
        dim_idx = batch_remainder // feature_dim
        feat_idx = batch_remainder % feature_dim
        
        # Process first output (first element from split)
        # This output is [batch_size, 1, feature_dim]
        out_first_offset = batch_idx * feature_dim + feat_idx
        mask_first = (batch_idx < batch_size) & (dim_idx == 0) & (feat_idx < feature_dim)
        
        # Load input and store to first output
        input_val = tl.load(in_ptr + out_first_offset, mask=mask_first, other=0.0)
        tl.store(out_first_ptr + out_first_offset, input_val, mask=mask_first)
        
        # Process second output (transposed and reshaped spatial data)
        # This output is [batch_size, feature_dim, spatial_dim, spatial_dim]
        # We need to transpose from [batch, spatial_dim, feature] to [batch, feature, spatial_dim, spatial_dim]
        if dim_idx >= 1:  # Only process part after split
            spatial_idx = dim_idx - 1  # Remove the first dimension (split)
            if spatial_idx < SPATIAL_SIZE:
                spatial_actual_dim = int(math.sqrt(SPATIAL_SIZE))
                out_second_offset = (
                    batch_idx * feature_dim * SPATIAL_SIZE +
                    feat_idx * SPATIAL_SIZE +
                    spatial_idx // spatial_actual_dim * spatial_actual_dim +  # spatial_h
                    spatial_idx % spatial_actual_dim  # spatial_w
                )
                input_val = tl.load(in_ptr + batch_idx * total_dim * feature_dim + 
                                   dim_idx * feature_dim + feat_idx, 
                                   mask=(batch_idx < batch_size) & (spatial_idx < SPATIAL_SIZE) & (feat_idx < feature_dim), 
                                   other=0.0)
                tl.store(out_second_ptr + out_second_offset, input_val, mask=(batch_idx < batch_size) & (spatial_idx < SPATIAL_SIZE) & (feat_idx < feature_dim))
    
    # Perform addition first
    added_tensor = in_0 + in_1
    
    batch_size = added_tensor.shape[0]
    total_dim = added_tensor.shape[1]
    feature_dim = added_tensor.shape[2]
    
    # Determine spatial dimension (remove first element for the spatial part)
    spatial_size = total_dim - 1
    assert spatial_size > 0 and int(math.sqrt(spatial_size)) ** 2 == spatial_size, f"Spatial dimension {spatial_size} must be a perfect square"
    
    # Allocate output tensors
    out_first = torch.empty([batch_size, 1, feature_dim], dtype=added_tensor.dtype, device=added_tensor.device)
    out_second = torch.empty([batch_size, feature_dim, int(math.sqrt(spatial_size)), int(math.sqrt(spatial_size))], dtype=added_tensor.dtype, device=added_tensor.device)
    
    # Choose optimal block size and launch grid
    total_elements = batch_size * total_dim * feature_dim
    
    # Use different configurations based on tensor sizes
    if feature_dim <= 384 and spatial_size <= 576:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
        
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_split_transpose_view_kernel[(num_programs,)](
        in_ptr=added_tensor,
        out_first_ptr=out_first,
        out_second_ptr=out_second,
        batch_size=batch_size,
        total_dim=total_dim,
        feature_dim=feature_dim,
        spatial_dim=int(math.sqrt(spatial_size)),
        SPATIAL_SIZE=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_first, out_second

def replacement_func():
    return fused_split_transpose_view