import torch
import triton
import triton.language as tl

def pattern(x, view_dim1=1):
    # Hardtanh activation
    tmp_0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    
    # Adaptive average pooling to (1, 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    
    # View operation with variable first dimension
    tmp_2 = tmp_1.view(view_dim1, -1)
    
    # Flatten from dimension 1
    tmp_3 = torch.flatten(tmp_2, 1)
    
    return tmp_3

def replacement_args(x, view_dim1):
    return (x, view_dim1)

@triton.jit
def fused_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    features,
    height,
    width,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature for one batch element (simpler, more direct approach)
    batch_idx = tl.program_id(1)  # batch dimension
    feat_idx = tl.program_id(0)  # feature index
    
    # Only process if feature index is within bounds
    if feat_idx >= features:
        return
        
    # Calculate starting index for this feature across all spatial locations
    spatial_start_idx = batch_idx * features * height * width + feat_idx * height * width
    
    # Load all spatial values for this feature and compute sum
    spatial_sum = 0.0
    
    # Simple loop for spatial locations (most straightforward approach)
    for i in range(height * width):
        spatial_idx = spatial_start_idx + i
        if spatial_idx < (batch_size * features * height * width):
            x_val = tl.load(x_ptr + spatial_idx)
            # Apply hardtanh: max(0, min(6, x))
            clamped_val = tl.maximum(tl.minimum(x_val, 6.0), 0.0)
            spatial_sum += clamped_val
    
    # Compute average
    total_spatial = height * width
    avg_value = spatial_sum / total_spatial
    
    # Store result
    out_idx = batch_idx * out_features + feat_idx
    if out_idx < (batch_size * out_features):
        tl.store(out_ptr + out_idx, avg_value)

@torch.fx.wrap
def fused_adaptive_pool_flatten(x, view_dim1):
    # Get input shape and determine output characteristics
    batch_size, features, height, width = x.shape
    
    # For adaptive_avg_pool2d to (1,1), height and width become 1
    # So after pooling: [batch, features, 1, 1]
    # Then view(view_dim1, -1): 
    #   - Total elements after pooling: batch * features * 1 * 1 = batch * features
    #   - view(view_dim1, -1) gives [batch, view_dim1, -1] 
    #   - The last dim is calculated as: (features * 1 * 1) // view_dim1 = features // view_dim1
    #   - So result is [batch, view_dim1, features // view_dim1]
    # Then flatten from dim 1: [batch, view_dim1 * features // view_dim1] = [batch, features]
    
    # So regardless of view_dim1, the final output should always be [batch_size, features]
    # This is because flatten from dim 1 combines the view_dim1 and features//view_dim1 dimensions
    out_features = features
    
    out_shape = (batch_size, out_features)
    
    # Allocate output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Use simplified grid configuration - one program per feature per batch
    grid = (features, batch_size)  # (features, batch_size)
    
    fused_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        features=features,
        height=height,
        width=width,
        out_features=out_features,
        BLOCK_SIZE=1,  # Not used in simplified kernel
    )
    
    return out

def replacement_func():
    return fused_adaptive_pool_flatten