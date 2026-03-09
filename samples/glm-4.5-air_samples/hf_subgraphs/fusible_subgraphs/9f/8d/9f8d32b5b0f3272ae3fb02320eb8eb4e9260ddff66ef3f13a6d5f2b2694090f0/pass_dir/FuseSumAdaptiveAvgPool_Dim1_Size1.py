import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the computation in model.py
def pattern(in_0):
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized fused kernel using Triton
@triton.jit
def fused_sum_adaptive_avg_pool_kernel(
    input_ptr,
    output_ptr,
    n_batch: tl.constexpr,
    n_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    depth: tl.constexpr,
):
    # Each program handles one (batch, feature) pair
    batch_id = tl.program_id(0)
    feature_id = tl.program_id(1)  # feature_id goes from 0 to 127 for 128 features
    
    # Compute total spatial size
    spatial_size = height * width * depth
    
    # For sum(dim=1): accumulate sum over input channel dimension (size 2)
    # For adaptive_avg_pool2d(..., 1): divide by total spatial size to get average
    # Only process valid features (0-127)
    if feature_id >= 128:
        return  # Early return for invalid features
        
    # Sum over input channels and spatial locations for this feature
    total_sum = 0.0
    
    for in_channel in range(0, 2):  # Only 2 input channels to sum over
        for spatial_offset in range(0, spatial_size):
            # Compute spatial coordinates
            h = spatial_offset // (width * depth)
            w = (spatial_offset % (width * depth)) // depth
            d = spatial_offset % depth
            
            # Bounds checking - avoid chained boolean operators by nesting ifs
            if h < height:
                if w < width:
                    if d < depth:
                        # Compute input index: [batch, in_channel, feature_id, h, w, d]
                        # Note: This assumes the input tensor has shape [B, 2, 128, H, W, D]
                        input_idx = (batch_id * 2 + in_channel) * 128 * spatial_size + \
                                   feature_id * spatial_size + spatial_offset
                        
                        # Load input element
                        x = tl.load(input_ptr + input_idx)
                        
                        # Accumulate sum
                        total_sum += x
    
    # Compute final average: divide by number of input channels (2) and spatial locations
    spatial_avg = total_sum / (2 * spatial_size)
    
    # Store output at [batch, feature_id] position
    output_idx = batch_id * 128 + feature_id
    tl.store(output_ptr + output_idx, spatial_avg)

# Wrapper function decorated with @torch.fx.wrap
@torch.fx.wrap
def fused_sum_adaptive_avg_pool(input_tensor):
    # Get input tensor properties
    shape = input_tensor.shape
    n_batch, n_channels_in, height, width, depth = shape
    
    # Set Triton kernel launch configuration - one program per (batch, feature) pair
    # Output after sum(dim=1) and adaptive_avg_pool2d should have shape [B, 128, 1, 1]
    num_programs_x = n_batch if n_batch > 0 else 1
    num_programs_y = 128 if 128 > 0 else 1  # 128 features from input
    
    # Prepare output tensor with correct shape [batch, 128, 1, 1]
    output_tensor = torch.empty((n_batch, 128, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch the Triton kernel with 2D grid
    fused_sum_adaptive_avg_pool_kernel[(num_programs_x, num_programs_y)](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_batch=n_batch,
        n_channels=n_channels_in,  # Should be 2
        height=height,
        width=width,
        depth=depth,
    )
    
    return output_tensor

# Replacement function that returns the kernel wrapper
def replacement_func():
    return fused_sum_adaptive_avg_pool