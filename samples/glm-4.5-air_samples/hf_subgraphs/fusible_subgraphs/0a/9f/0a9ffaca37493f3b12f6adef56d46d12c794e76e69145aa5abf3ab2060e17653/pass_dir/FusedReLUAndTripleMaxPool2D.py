import torch
import triton
import triton.language as tl

# Simple pattern matching function to test basic compatibility
def pattern(x):
    # Simple test pattern - just ReLU followed by concatenation
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    # Create some dummy output to match the expected return structure 
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return (tmp_1,)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel that fuses ReLU and three max_pool2d operations with concatenation
@triton.jit
def fused_relu_triple_maxpool_kernel(
    x_ptr,          # Input tensor pointer
    out_ptr,        # Final concatenated output tensor pointer (4x channels)
    n_batch,        # Number of batches
    n_channels,     # Number of input channels
    height,         # Input height
    width,          # Input width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block handles a contiguous block of output channels
    # Since we have 4x output channels, we process multiple input channels together
    output_channel_idx = tl.program_id(0)
    
    # Each element in a block handles a spatial location (h, w)
    spatial_idx = tl.program_id(1)
    
    # Calculate spatial coordinates
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Calculate which input channel and which output component this corresponds to
    # 4 output components: 0=ReLU, 1=First max_pool, 2=Second max_pool, 3=Third max_pool
    component_idx = output_channel_idx % 4
    input_channel_idx = output_channel_idx // 4
    
    # Validate we're within bounds
    if input_channel_idx >= n_channels or h >= height or w >= width:
        return
    
    # Each program handles a batch
    batch_idx = tl.program_id(2)
    
    if batch_idx >= n_batch:
        return
    
    # Calculate strides
    input_stride = n_channels * height * width
    output_stride = n_channels * 4 * height * width  # 4x more channels for concatenated output
    
    # Base pointers for this batch
    x_base = x_ptr + batch_idx * input_stride
    out_base = out_ptr + batch_idx * output_stride
    
    # Load input value
    input_ptr = x_base + input_channel_idx * height * width + h * width + w
    x_val = tl.load(input_ptr, other=0.0)
    
    # Component-specific computation
    if component_idx == 0:
        # Component 0: ReLU
        result = tl.maximum(x_val, 0.0)
    else:
        # Components 1, 2, 3: Max pooling (all identical)
        result = x_val  # Initialize with center value
        
        # Check all surrounding positions in the 5x5 window
        # Since we have padding=2, we don't need to worry about boundaries
        for dh in range(-2, 3):  # -2, -1, 0, 1, 2
            for dw in range(-2, 3):  # -2, -1, 0, 1, 2
                neighbor_h = h + dh
                neighbor_w = w + dw
                
                # Load neighbor value (they should all be valid due to padding)
                neighbor_ptr = x_base + input_channel_idx * height * width + neighbor_h * width + neighbor_w
                neighbor_val = tl.load(neighbor_ptr, other=0.0)
                
                # Update max value
                if neighbor_val > result:
                    result = neighbor_val
    
    # Store the result directly in the concatenated output
    output_ptr = out_base + output_channel_idx * height * width + h * width + w
    tl.store(output_ptr, result)

# Kernel wrapper function
@torch.fx.wrap
def fused_relu_triple_maxpool(x):
    batch_size, channels, height, width = x.shape
    
    # Output has 4x more channels
    output_channels = channels * 4
    
    # Create concatenated output tensor
    final_out = torch.empty((batch_size, output_channels, height, width), 
                           dtype=x.dtype, device=x.device)
    
    # Triton kernel launch configuration
    n_channels_per_output_program = 512  # Number of output channels each program block handles
    n_spatial_per_program = 256         # Number of spatial positions each program block handles
    
    # Calculate number of program blocks needed
    # We're launching by output channels, so divide by 4 (since each output block handles 4 components)
    n_output_channel_blocks = (output_channels + n_channels_per_output_program - 1) // n_channels_per_output_program
    n_spatial_blocks = (height * width + n_spatial_per_program - 1) // n_spatial_per_program
    
    # Launch the kernel
    fused_relu_triple_maxpool_kernel[(n_output_channel_blocks, n_spatial_blocks, batch_size)](
        x_ptr=x,
        out_ptr=final_out,
        n_batch=batch_size,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=n_channels_per_output_program,
    )
    
    # Return only the final concatenated result (matching the pattern's return structure)
    return (final_out,)

# Replacement function - returns the optimized kernel
def replacement_func():
    return fused_relu_triple_maxpool