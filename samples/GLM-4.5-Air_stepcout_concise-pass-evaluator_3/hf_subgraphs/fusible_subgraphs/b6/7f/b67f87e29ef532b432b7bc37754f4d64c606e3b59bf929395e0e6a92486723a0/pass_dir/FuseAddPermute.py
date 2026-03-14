import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match the addition + permutation pattern:
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    Return the permuted result
    """
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    return tmp_1

def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement
    We need the input tensors and their shapes
    """
    return (in_0, in_1)

@triton.jit
def fused_add_permute_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_batch,
    n_features,
    n_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that adds two tensors and permutes the last two dimensions.
    Input: [batch, features, spatial] 
    Output: [batch, spatial, features]
    
    Optimized version with better memory coalescing.
    """
    # Calculate program indices
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1) 
    spatial_offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Early check for out-of-bounds spatial indices
    spatial_mask = spatial_offset < n_spatial
    
    # Calculate input offset for this feature and spatial positions
    input_offset_base = batch_idx * n_features * n_spatial + feature_idx * n_spatial
    input_offset = input_offset_base + spatial_offset
    
    # Load input data with contiguous spatial access
    x = tl.load(x_ptr + input_offset, mask=spatial_mask, other=0.0)
    y = tl.load(y_ptr + input_offset, mask=spatial_mask, other=0.0)
    
    # Compute element-wise addition
    out = x + y
    
    # Calculate output offset in permuted layout
    # Each program handles one feature, but writes to scattered spatial locations
    output_offset_base = batch_idx * n_spatial * n_features
    # Each spatial position gets all features, so we interleave features
    output_offset = output_offset_base + spatial_offset * n_features + feature_idx
    
    # Store result in permuted layout
    tl.store(out_ptr + output_offset, out, mask=spatial_mask)

@torch.fx.wrap
def fused_add_permute(x, y):
    """
    Wrapper function for the fused add+permute kernel
    """
    # Get input tensor shapes
    batch_size, n_features, n_spatial = x.shape
    
    # Launch grid configuration - one program per feature
    BLOCK_SIZE = 1024
    grid_z = batch_size
    grid_y = n_features  # One program per feature
    grid_x = (n_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE  # Programs per spatial dimension
    
    # Create output tensor with permuted layout
    out = torch.empty(batch_size, n_spatial, n_features, dtype=x.dtype, device=x.device)
    
    # Launch the kernel
    fused_add_permute_kernel[(grid_z, grid_y, grid_x)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_batch=batch_size,
        n_features=n_features,
        n_spatial=n_spatial,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """
    Return the fused kernel function
    """
    return fused_add_permute