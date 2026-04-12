import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match the conv2d + batch_norm pattern with float32 data type
    """
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.avg_pool2d(in_5, 2, 2, 0, True, False, None)
    return (tmp_7, tmp_6)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

# Triton kernel for fused conv2d + batch norm with float32
@triton.jit
def fused_conv2d_batch_norm_kernel_fp32(
    input_ptr, weight_ptr, bias_ptr, 
    running_mean_ptr, running_var_ptr, weight_norm_ptr,
    output_ptr, n_channels, height, width,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    kernel_h: tl.constexpr, kernel_w: tl.constexpr,
    eps: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate output positions
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    spatial_idx = pid % (height * width)
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Calculate input position with padding
    in_h = h * stride_h - pad_h
    in_w = w * stride_w - pad_w
    
    # Initialize accumulator with higher precision for float32
    acc = 0.0
    
    # Convolution computation optimized for float32
    for c_in in range(n_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute input coordinates
                src_h = in_h + kh
                src_w = in_w + kw
                
                # Skip if out of bounds (equivalent to padding)
                if src_h < 0 or src_h >= height or src_w < 0 or src_w >= width:
                    continue
                
                # Load input and weight with float32 support
                input_idx = batch_idx * n_channels * height * width + c_in * height * width + src_h * width + src_w
                weight_idx = c_in * kernel_h * kernel_w + kh * kernel_w + kw
                
                input_val = tl.load(input_ptr + input_idx)
                weight_val = tl.load(weight_ptr + weight_idx)
                
                acc += input_val * weight_val
    
    # Load bias and normalization parameters
    bias = tl.load(bias_ptr + batch_idx * n_channels)
    running_mean = tl.load(running_mean_ptr + batch_idx * n_channels)
    running_var = tl.load(running_var_ptr + batch_idx * n_channels)
    weight_norm = tl.load(weight_norm_ptr + batch_idx * n_channels)
    
    # Batch normalization with optimized math for float32
    normalized_acc = (acc - running_mean) * (weight_norm / tl.sqrt(running_var + eps)) + bias
    
    # Store result
    output_idx = batch_idx * n_channels * height * width + spatial_idx
    tl.store(output_ptr + output_idx, normalized_acc)

# Optimized average pooling kernel for float32
@triton.jit
def avg_pool2d_kernel_fp32(
    input_ptr, output_ptr, 
    n_channels, in_height, in_width, out_height, out_width,
    pool_size: tl.constexpr, stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (out_height * out_width)
    spatial_idx = pid % (out_height * out_width)
    h = spatial_idx // out_width
    w = spatial_idx % out_width
    
    # Calculate input window region with optimized boundary checking
    in_h_start = h * stride
    in_h_end = min(in_h_start + pool_size, in_height)
    in_w_start = w * stride  
    in_w_end = min(in_w_start + pool_size, in_width)
    
    # Average pooling with accumulation optimization for float32
    sum_val = 0.0
    count = 0
    
    for c in range(n_channels):
        for kh in range(in_h_start, in_h_end):
            for kw in range(in_w_start, in_w_end):
                input_idx = batch_idx * n_channels * in_height * in_width + c * in_height * in_width + kh * in_width + kw
                input_val = tl.load(input_ptr + input_idx)
                sum_val += input_val
                count += 1
    
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    output_idx = batch_idx * n_channels * out_height * out_width + spatial_idx
    tl.store(output_ptr + output_idx, avg_val)

# Kernel wrapper for fused conv2d + batch norm (float32)
@torch.fx.wrap
def fused_conv2d_batch_norm_fp32(in_6, in_4, in_2, in_0, in_1, in_3):
    # Get tensor shapes
    batch_size, in_channels, in_height, in_width = in_6.shape
    out_channels, _, kernel_h, kernel_w = in_4.shape
    
    # Calculate output dimensions
    out_height = in_height
    out_width = in_width
    
    # Allocate output tensor with float32
    output = torch.empty((batch_size, out_channels, out_height, out_width), 
                       dtype=torch.float32, device=in_6.device)
    
    # Launch kernel with optimized grid size for float32
    grid_size = batch_size * out_height * out_width
    fused_conv2d_batch_norm_kernel_fp32[grid_size // 2048 + (1 if grid_size % 2048 else 0), 1, 1](
        in_6, in_4, in_2, in_0, in_1, in_3,
        output, out_channels, out_height, out_width,
        1, 1,  # stride
        1, 1,  # padding  
        kernel_h, kernel_w,  # kernel size
        1e-5, 256  # eps, optimized block size for float32
    )
    
    return output

# Kernel wrapper for avg_pool2d (float32)
@torch.fx.wrap
def optimized_avg_pool2d_fp32(in_5):
    # Get tensor shapes
    batch_size, channels, in_height, in_width = in_5.shape
    
    # Calculate output dimensions
    pool_size = 2
    stride = 2
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1
    
    # Allocate output tensor with float32
    output = torch.empty((batch_size, channels, out_height, out_width), 
                       dtype=torch.float32, device=in_5.device)
    
    # Launch kernel with optimized grid size
    grid_size = batch_size * out_height * out_width
    avg_pool2d_kernel_fp32[grid_size // 1024 + (1 if grid_size % 1024 else 0), 1, 1](
        in_5, output,
        channels, in_height, in_width, out_height, out_width,
        pool_size, stride, 256  # optimized block size for float32
    )
    
    return output

# Replacement function for float32
def replacement_func():
    def optimized_kernel_fp32(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        # Fused conv2d + batch norm
        batch_norm_output = fused_conv2d_batch_norm_fp32(in_6, in_4, in_2, in_0, in_1, in_3)
        
        # Average pooling
        pool_output = optimized_avg_pool2d_fp32(in_5)
        
        return (pool_output, batch_norm_output)
    
    return optimized_kernel_fp32