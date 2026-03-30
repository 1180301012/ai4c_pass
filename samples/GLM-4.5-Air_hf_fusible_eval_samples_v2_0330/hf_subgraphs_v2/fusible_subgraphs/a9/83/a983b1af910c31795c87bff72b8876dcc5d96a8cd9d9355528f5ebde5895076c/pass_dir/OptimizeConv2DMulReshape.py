import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias):
    """Match Conv2D -> multiply by 1.0 -> reshape pattern"""
    # Conv2D with fixed parameters: stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    conv_output = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    # Multiplication by 1.0 is a no-op that we can eliminate
    mul_output = conv_output * 1.0
    # Reshape to (-1, 17, 4096)
    reshape_output = mul_output.reshape(-1, 17, 4096)
    return reshape_output

def replacement_args(conv_input, conv_weight, conv_bias):
    """Extract arguments for optimized kernel"""
    return (conv_input, conv_weight, conv_bias)

@triton.jit
def simple_1x1_conv_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    batch_size: tl.constexpr,
    out_channels: tl.constexpr,
    in_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple 1x1 convolution kernel that eliminates no-op operations"""
    pid = tl.program_id(0)
    
    # Each thread handles one output element: [b, c, h, w]
    # Calculate which output element this thread handles
    spatial_size = height * width
    total_elements = batch_size * out_channels * spatial_size
    
    if pid >= total_elements:
        return
    
    # Decode position from linear index
    spatial_idx = pid % spatial_size
    c = (pid // spatial_size) % out_channels
    b = pid // (out_channels * spatial_size)
    
    # Calculate spatial coordinates
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Compute 1x1 convolution: sum over input channels
    acc = 0.0
    
    # Load bias
    bias_val = tl.load(b_ptr + c)
    
    # Accumulate over input channels
    for k in range(0, in_channels, BLOCK_SIZE):
        k_end = min(k + BLOCK_SIZE, in_channels)
        
        # Calculate memory offsets
        x_offset = (b * in_channels + k) * height * width + h * width + w
        w_offset = (c * in_channels + k)
        
        # Load with bounds checking
        x_val = tl.load(x_ptr + x_offset, mask=(k < in_channels), other=0.0)
        w_val = tl.load(w_ptr + w_offset, mask=(k < in_channels), other=0.0)
        
        acc += x_val * w_val
    
    # Store result
    output_offset = (b * out_channels + c) * height * width + h * width + w
    tl.store(y_ptr + output_offset, acc + bias_val)

@torch.fx.wrap  
def optimized_conv_reshape(conv_input, conv_weight, conv_bias):
    """Wrapper function that eliminates no-op multiplication and optimizes conv+reshape"""
    # Get tensor dimensions
    batch_size, in_channels, height, width = conv_input.shape
    out_channels, _, _, _ = conv_weight.shape
    
    # For 1x1 convolution with stride 1, spatial dimensions remain the same
    output = torch.empty((batch_size, out_channels, height, width), 
                       dtype=conv_input.dtype, 
                       device=conv_input.device)
    
    # Configure block size for efficient memory access
    BLOCK_SIZE = 32
    
    # Calculate grid size for 1D launch
    # Each thread computes one output element
    spatial_size = height * width
    total_elements = batch_size * out_channels * spatial_size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized 1x1 convolution kernel
    simple_1x1_conv_kernel[(grid_size,)](
        conv_input,
        conv_weight,
        conv_bias,
        output,
        batch_size,
        out_channels,
        in_channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape to (-1, 17, 4096) - this operation stays the same
    return output.reshape(-1, 17, 4096)

def replacement_func():
    """Return the optimized function"""
    return optimized_conv_reshape