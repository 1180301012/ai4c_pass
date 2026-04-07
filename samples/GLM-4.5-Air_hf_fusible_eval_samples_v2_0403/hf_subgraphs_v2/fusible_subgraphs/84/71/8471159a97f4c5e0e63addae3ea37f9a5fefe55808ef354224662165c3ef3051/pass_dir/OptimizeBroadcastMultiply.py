import torch
import triton
import triton.language as tl

def pattern(sigmoid_output, feature_map):
    """
    Pattern matches: view operation followed by element-wise multiplication
    This pattern appears in all the target graphs as:
    tmp_4 = tmp_3.view(shape)  # where shape includes 1s for spatial dimensions
    tmp_5 = in_3 * tmp_4  # broadcast multiplication
    """
    # The view operation reshapes the sigmoid output to have spatial dimensions of 1
    # for broadcasting across the feature map
    tmp_4 = sigmoid_output.view(sigmoid_output.shape[0], sigmoid_output.shape[1], 1, 1)
    tmp_5 = feature_map * tmp_4
    return tmp_5

def replacement_args(sigmoid_output, feature_map):
    """Extract arguments for the broadcast multiplication kernel"""
    return (sigmoid_output, feature_map)

@triton.jit
def optimized_broadcast_multiply_kernel(
    sigmoid_ptr,        # Pointer to sigmoid output [batch, features, 1, 1]
    feature_ptr,        # Pointer to feature map [batch, features, H, W]
    output_ptr,         # Pointer to output [batch, features, H, W]
    batch_size,         # Batch size
    features,           # Number of features
    height,             # Spatial height
    width,              # Spatial width
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID handles spatial tiles
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    batch_id = tl.program_id(2)
    
    # Calculate spatial tile range
    h_start = pid_h * BLOCK_SIZE
    w_start = pid_w * BLOCK_SIZE
    
    # Offsets for spatial tile (with padding protection)
    h_offset = tl.arange(0, BLOCK_SIZE) + h_start
    w_offset = tl.arange(0, BLOCK_SIZE) + w_start
    
    # Mask to ensure we don't go out of bounds
    h_mask = h_offset < height
    w_mask = w_offset < width
    
    # Load sigmoid values (broadcast across spatial dimensions)
    sigmoid_vals = tl.load(
        sigmoid_ptr + batch_id * features + tl.arange(0, features),
        mask=tl.arange(0, features) < features,
        other=0.0
    )
    
    # Load feature values for this spatial tile - simplify memory access
    spatial_offset = h_offset[:, None] * width + w_offset[None, :]
    feature_offset = tl.arange(0, features)[..., None, None] * height * width
    total_offset = spatial_offset + feature_offset
    
    feature_vals = tl.load(
        feature_ptr + batch_id * features * height * width + total_offset,
        mask=(h_mask[:, None] & w_mask[None, :] & (tl.arange(0, features)[..., None, None] < features)),
        other=0.0
    )
    
    # Apply sigmoid attention weights ( broadcasting across spatial dimensions)
    sigmoid_4d = sigmoid_vals[None, None, :].to(dtype=tl.float32)
    feature_4d = feature_vals.to(dtype=tl.float32)
    output_4d = feature_4d * sigmoid_4d
    
    # Store results - simplify memory access
    spatial_offset = (h_offset[:, None] * width + w_offset[None, :])[..., None, None]
    feature_offset = tl.arange(0, features)[..., None, None, None]
    total_offset = spatial_offset * height * width + feature_offset
    
    tl.store(
        output_ptr + batch_id * features * height * width + total_offset,
        output_4d,
        mask=(h_mask[:, None] & w_mask[None, :] & (tl.arange(0, features)[..., None, None, None] < features))
    )

@torch.fx.wrap
def optimized_broadcast_multiply_torch(sigmoid_output, feature_map):
    """Torch wrapper for optimized broadcast multiplication kernel"""
    # Handle different input shapes - sigmoid_output should be [batch, features]
    # feature_map should be [batch, features, H, W]
    if sigmoid_output.dim() != 2:
        # Handle case where sigmoid_output is already shaped with spatial dims
        print("Warning: sigmoid_output already has spatial dimensions, using as-is")
        return feature_map * sigmoid_output
    
    batch_size, features = sigmoid_output.shape
    _, _, height, width = feature_map.shape
    
    # Create output tensor with same dtype as input
    output = torch.empty_like(feature_map)
    
    # Set kernel parameters - tune based on GPU architecture
    BLOCK_SIZE = 32  # Spatial tile size
    
    # Calculate grid dimensions
    num_blocks_h = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_w = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks_h, num_blocks_w, batch_size)
    
    # Launch kernel
    optimized_broadcast_multiply_kernel[grid](
        sigmoid_ptr=sigmoid_output,
        feature_ptr=feature_map,
        output_ptr=output,
        batch_size=batch_size,
        features=features,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized broadcast multiplication function"""
    return optimized_broadcast_multiply_torch