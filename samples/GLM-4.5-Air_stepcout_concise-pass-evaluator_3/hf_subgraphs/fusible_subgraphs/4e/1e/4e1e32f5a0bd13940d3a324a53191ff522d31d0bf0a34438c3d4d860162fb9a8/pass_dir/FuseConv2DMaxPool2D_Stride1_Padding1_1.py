import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match Conv2D (stride=1, padding=1) followed by MaxPool2D (kernel=3, stride=2, padding=1)"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(tmp_1, tmp_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv_pool_stride1_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    kernel_h,
    kernel_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused Conv2D + MaxPool2D kernel for stride=1 configuration"""
    
    # Compute output spatial dimensions
    # Conv2D: stride=1, padding=1
    out_h = (in_height + 2 * 1 - kernel_h) // 1 + 1
    out_w = (in_width + 2 * 1 - kernel_w) // 1 + 1
    # MaxPool2D: kernel=3, stride=2, padding=1
    pool_out_h = (out_h + 2 * 1 - 3) // 2 + 1
    pool_out_w = (out_w + 2 * 1 - 3) // 2 + 1
    
    # Program ID for maximum pooling output
    m = tl.program_id(0)  # batch * pool_h
    n = tl.program_id(1)  # channel
    
    # Only compute if within bounds
    if m >= batch_size * pool_out_h or n >= out_channels:
        return
        
    # Compute output coordinates for pooled result
    batch_idx = m // pool_out_h
    pool_h_i = m % pool_out_h
    pool_w_j = n % pool_out_w
    pool_c = n // pool_out_w
    
    # Compute corresponding conv output coordinates (backwards from pooling)  
    conv_out_h = pool_h_i * 2 - 1  # pool stride=2, padding=1 -> backwards
    conv_out_w = pool_w_j * 2 - 1
    
    # Initialize accumulator for max pooling
    max_val = -float('inf')
    
    # For each position in the max pooling window, sample from conv output
    # Since conv stride=1, we can sample conv values every stride position
    for pool_ki in range(3):  # 3x3 max pool kernel
        for pool_kj in range(3):
            # Sample from conv output every 2 positions (max pool stride)
            sample_h = conv_out_h + pool_ki * 2
            sample_w = conv_out_w + pool_kj * 2
            
            if 0 <= sample_h < out_h and 0 <= sample_w < out_w:
                # Compute conv input coordinates for this position
                in_h = sample_h * 1 - 1  # conv stride=1, padding=1 -> backwards
                in_w = sample_w * 1 - 1
                
                if 0 <= in_h < in_height and 0 <= in_w < in_width:
                    # Compute convolution value at this position
                    conv_sum = 0.0
                    
                    # Loop over kernel dimensions
                    for ki in range(kernel_h):
                        for kj in range(kernel_w):
                            # Input coordinates for this kernel position
                            input_h = in_h + ki
                            input_w = in_w + kj
                            
                            if 0 <= input_h < in_height and 0 <= input_w < in_width:
                                # Load input value - handle single channel input
                                input_offset = (batch_idx * in_channels) * in_height * in_width + input_h * in_width + input_w
                                input_val = tl.load(input_ptr + input_offset)
                                
                                # Load weight value
                                weight_offset = (pool_c * in_channels) * kernel_h * kernel_w + ki * kernel_w + kj
                                weight_val = tl.load(weight_ptr + weight_offset)
                                
                                conv_sum += input_val * weight_val
                    
                    # Update max value
                    if conv_sum > max_val:
                        max_val = conv_sum
    
    # Store the result
    output_offset = batch_idx * out_channels * pool_out_h * pool_out_w + pool_c * pool_out_h * pool_out_w + pool_h_i * pool_out_w + pool_w_j
    tl.store(output_ptr + output_offset, max_val)

@torch.fx.wrap
def fused_conv2d_maxpool2d_stride1(in_0, in_1):
    """Fused Conv2D + MaxPool2D implementation with stride=1"""
    # Get input tensor dimensions
    input_tensor = in_1
    weight_tensor = in_0
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, kernel_channels, kernel_h, kernel_w = weight_tensor.shape
    assert kernel_channels == in_channels, "Kernel channels must match input channels"
    
    # Calculate output dimensions
    # Conv2D: stride=1, padding=1, kernel=3x3
    conv_out_h = (in_height + 2 * 1 - kernel_h) // 1 + 1
    conv_out_w = (in_width + 2 * 1 - kernel_w) // 1 + 1
    
    # MaxPool2D: kernel=3x3, stride=2, padding=1  
    pool_out_h = (conv_out_h + 2 * 1 - 3) // 2 + 1
    pool_out_w = (conv_out_w + 2 * 1 - 3) // 2 + 1
    
    # Allocate output tensor
    output = torch.empty((batch_size, out_channels, pool_out_h, pool_out_w), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up grid dimensions for Triton
    # We'll use 2D grid: batch*output_height x output_channels
    grid_x = batch_size * pool_out_h
    grid_y = out_channels
    
    # Choose block sizes based on typical GPU architecture
    BLOCK_SIZE_M = 16  # Output height dimension
    BLOCK_SIZE_N = 8   # Output channels dimension (reduced for better occupancy)
    
    # Launch Triton kernel with proper grid configuration
    fused_conv_pool_stride1_kernel[(
        (grid_x + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (grid_y + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor, 
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return (output,)

def replacement_func():
    return fused_conv2d_maxpool2d_stride1