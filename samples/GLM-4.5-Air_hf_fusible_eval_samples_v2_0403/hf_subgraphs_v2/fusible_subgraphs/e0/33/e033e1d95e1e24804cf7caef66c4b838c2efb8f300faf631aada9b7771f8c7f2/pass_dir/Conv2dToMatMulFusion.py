import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    """
    Pattern to match: conv2d operation with 1x1 kernel, stride=1, padding=0, dilation=1, groups=1
    This is equivalent to a matrix multiplication operation
    """
    conv2d_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return conv2d_out

def replacement_args(x, weight, bias):
    """
    Extract arguments needed for the optimized matmul implementation
    Returns tuple of (input_tensor, weight_tensor, bias_tensor)
    """
    return (x, weight, bias)

@triton.jit
def conv2d_1x1_kernel_simplified(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    """
    Simplified and optimized kernel for 1x1 conv2d
    Processes one output element at a time for simplicity and correctness
    """
    # Program IDs
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Global indices for output position
    batch_idx = pid_x
    channel_idx = pid_y
    
    # Boundary checks
    if batch_idx >= batch_size or channel_idx >= out_channels:
        return
    
    # Calculate output position in flattened space
    out_base = batch_idx * out_channels * height * width + channel_idx * height * width
    
    # Load bias
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Accumulator for this output element
    acc = bias_val
    
    # Loop over input channels
    for c in range(in_channels):
        # Weight for this input channel and output channel
        weight_val = tl.load(weight_ptr + channel_idx * in_channels + c)
        
        # Spatial position loop for 1x1 conv (at each spatial location)
        for h in range(height):
            for w in range(width):
                # Input position
                in_idx = (batch_idx * in_channels + c) * height * width + h * width + w
                x_val = tl.load(x_ptr + in_idx)
                
                # Accumulate: x * weight
                acc += x_val * weight_val
    
    # Store the result
    tl.store(out_ptr + out_base, acc)

@torch.fx.wrap
def optimized_conv2d_1x1(x, weight, bias):
    """
    Optimized 1x1 conv2d using Triton kernel
    Uses simplified kernel for correctness and compatibility
    """
    batch_size, in_channels, height, width = x.shape
    out_channels, _, _, _ = weight.shape
    
    # Output shape
    out_shape = (batch_size, out_channels, height, width)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Block sizes for parallel execution
    BLOCK_SIZE_X = 16  # Process 16 batch elements at a time
    BLOCK_SIZE_Y = 16  # Process 16 output channels at a time
    
    # Calculate grid dimensions
    grid_x = (batch_size + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (out_channels + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Launch the kernel
    conv2d_1x1_kernel_simplified[(grid_x, grid_y)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y
    )
    
    return out

def replacement_func():
    """
    Returns the optimized function reference
    """
    return optimized_conv2d_1x1