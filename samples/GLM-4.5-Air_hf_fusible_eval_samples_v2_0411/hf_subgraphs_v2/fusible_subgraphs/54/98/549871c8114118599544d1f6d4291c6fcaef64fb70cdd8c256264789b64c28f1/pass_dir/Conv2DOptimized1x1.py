import torch
import triton
import triton.language as tl
import math

# Pattern matching function - start with simple Conv2D
def pattern(in_0, in_1, in_2):
    # Simple Conv2D pattern first
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return (tmp_2,)

# Argument extraction function - correct argument order for Conv2D
def replacement_args(in_0, in_1, in_2):
    # in_0: bias, in_1: weight, in_2: input
    # Return in Conv2D order: (input, weight, bias)
    return (in_2, in_1, in_0)

# Simplified kernel that does full 1x1 convolution with SiLU activation
@triton.jit
def conv2d_silu_kernel(
    x_ptr,           # Input [N, C_in, H, W]
    w_ptr,           # Weight [C_out, C_in, 1, 1]
    b_ptr,           # Bias [C_out]
    out_ptr,         # Output [N, C_out, H, W]
    N, C_in, H, W, C_out,
):
    # Each kernel instance handles one output channel at one spatial position
    batch_idx = tl.program_id(0)
    spatial_pos = tl.program_id(1)
    c_out_idx = tl.program_id(2)
    
    # Calculate spatial position from flattened index
    h_idx = spatial_pos // W
    w_idx = spatial_pos % W
    
    # Initialize accumulator with bias
    bias_offset = c_out_idx
    bias_val = tl.load(b_ptr + bias_offset)
    if bias_val.dtype == tl.bfloat16 or bias_val.dtype == tl.float16:
        acc = bias_val.to(tl.float32)
    else:
        acc = bias_val
    
    # Process all input channels (simple loop, no tiling for now)
    for c_in_idx in range(C_in):
        # Load input value at this channel and spatial position
        input_offset = batch_idx * C_in * H * W + c_in_idx * H * W + h_idx * W + w_idx
        x_val = tl.load(x_ptr + input_offset)
        
        # Load weight for this input and output channel
        weight_offset = c_out_idx * C_in + c_in_idx
        w_val = tl.load(w_ptr + weight_offset)
        
        # Convert to float32 for computation if needed
        if x_val.dtype == tl.bfloat16 or x_val.dtype == tl.float16:
            x_val_f32 = x_val.to(tl.float32)
            w_val_f32 = w_val.to(tl.float32)
        else:
            x_val_f32 = x_val
            w_val_f32 = w_val
        
        # Add contribution from this channel
        acc += x_val_f32 * w_val_f32
    
    # Apply SiLU activation: x * sigmoid(x)
    out = acc / (1.0 + tl.exp(-acc))
    
    # Store result
    output_offset = batch_idx * C_out * H * W + c_out_idx * H * W + h_idx * W + w_idx
    
    # Load a sample to determine output dtype
    sample_offset = batch_idx * C_in * H * W + h_idx * W + w_idx
    sample_val = tl.load(x_ptr + sample_offset)
    if sample_val.dtype == tl.bfloat16 or sample_val.dtype == tl.float16:
        tl.store(out_ptr + output_offset, out.to(sample_val.dtype))
    else:
        tl.store(out_ptr + output_offset, out)

# Kernel wrapper
@torch.fx.wrap
def conv2d_silu_optimized(input, weight, bias):
    # Get tensor shapes
    N, C_in, H, W = input.shape
    C_out = weight.shape[0]
    
    # Allocate output tensor
    out = torch.empty((N, C_out, H, W), dtype=input.dtype, device=input.device)
    
    # Calculate grid dimensions:
    # - batch dimension: N (1 in our case)
    # - flattened spatial positions: H * W (4 * 256 = 1024)  
    # - output channels: C_out (256)
    grid_z = N
    grid_x = H * W  
    grid_y = C_out
    
    # Launch kernel with proper 3D grid configuration
    conv2d_silu_kernel[(grid_z, grid_x, grid_y)](
        input, weight, bias, out,
        N, C_in, H, W, C_out
    )
    
    return out

# Replacement function (MUST be zero-arg, return function ref)
def replacement_func():
    return conv2d_silu_optimized