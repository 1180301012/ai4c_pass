import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match Conv2D operation only"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_2, in_1, in_0)  # Return (input, weight, bias)

@triton.jit
def fused_conv2d_silu_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    """Optimized Fused Conv2D + SiLU kernel using Triton"""
    # 2D grid: x dimension for spatial positions, y dimension for output channels
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    # Total spatial positions
    total_spatial = H * W
    if pid_x >= total_spatial:
        return
    
    # Total output channels per batch
    total_channels = N * C_out
    if pid_y >= total_channels:
        return
    
    # Convert spatial index to [h, w]
    w = pid_x % W
    h = pid_x // W
    
    # Convert channel index to [n, c_out]
    c_out = pid_y % C_out
    n = pid_y // C_out
    
    # Process multiple channels per program for better efficiency
    start_channel = (pid_x * BLOCK_SIZE_Y) % C_out
    max_channel = min(start_channel + BLOCK_SIZE_Y, C_out)
    
    # Process each channel in this block
    for block_c_out in range(start_channel, max_channel):
        # Calculate offset for this channel
        output_offset = (n * C_out + block_c_out) * H * W + h * W + w
        
        # Handle bias for this output channel
        conv_val = 0.0
        if bias_ptr is not None:
            conv_val = tl.load(bias_ptr + block_c_out)
        
        # For 1x1 convolution: sum over input channels
        for c_in in range(C_in):
            # Load weight for [block_c_out, c_in]
            weight_offset = block_c_out * C_in + c_in
            weight = tl.load(weight_ptr + weight_offset)
            
            # Load input for [n, c_in, h, w] 
            input_offset = n * C_in * H * W + c_in * H * W + h * W + w
            input_val = tl.load(input_ptr + input_offset)
            
            conv_val += input_val * weight
        
        # Apply SiLU activation: x * sigmoid(x)
        exp_val = tl.exp(-tl.abs(conv_val))
        if conv_val > 0:
            silu_val = conv_val / (1.0 + exp_val)
        else:
            silu_val = conv_val * exp_val / (1.0 + exp_val)
        
        # Store the final result
        tl.store(output_ptr + output_offset, silu_val)

@torch.fx.wrap  
def fused_conv2d_silu(input_tensor, weight_tensor, bias_tensor):
    """Wrapper function for fused Conv2D + SiLU operation"""
    N, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]
    
    output_shape = (N, C_out, H, W)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use 2D grid: spatial positions and channels
    BLOCK_SIZE_X = 64  # Process multiple spatial positions per block
    BLOCK_SIZE_Y = 32  # Process multiple output channels per block
    
    # Number of programs needed for each dimension
    num_spatial_programs = (H * W + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    num_channel_programs = (N * C_out + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Launch the kernel with 2D grid
    fused_conv2d_silu_kernel[(num_spatial_programs, num_channel_programs)](
        input_tensor,
        weight_tensor, 
        bias_tensor,
        output,
        N, C_in, H, W, C_out,
        BLOCK_SIZE_X, BLOCK_SIZE_Y
    )
    
    return output

def replacement_func():
    """Return the fused kernel wrapper function"""
    return fused_conv2d_silu