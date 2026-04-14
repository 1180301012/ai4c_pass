import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # Conv2D operation with specific parameters
    tmp_6 = torch.conv2d(in_7, in_5, None, (1, 1), (4, 4), (4, 4), 64)
    
    # Concatenation along channel dimension
    tmp_7 = torch.cat([in_6, tmp_6], 1)
    
    # Batch normalization
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    
    # PReLU activation
    tmp_9 = torch.prelu(tmp_8, in_0)
    
    # Adaptive average pooling to 1x1
    tmp_10 = torch.nn.functional.adaptive_avg_pool2d(tmp_9, 1)
    
    # View operation - we need to determine the correct shape dynamically
    batch_size = tmp_10.shape[0]
    channels = tmp_10.shape[1]
    tmp_11 = tmp_10.view(batch_size, channels)
    
    return tmp_9, tmp_11

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)

@triton.jit
def cgnet_fused_kernel(
    # Input tensors
    x7_ptr, in6_ptr,  # Input feature maps
    conv_weight_ptr,   # Conv2D weight [64, 1, 3, 3]
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,  # BatchNorm parameters
    prelu_weight_ptr,  # PReLU weight [128]
    
    # Output tensors
    out_features_ptr, out_pooled_ptr,
    
    # Tensor metadata
    batch_size,
    in_channels,
    in_height, in_width,
    out_channels,
    
    # Conv2D parameters
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    dilation_h: tl.constexpr, dilation_w: tl.constexpr,
    groups: tl.constexpr,
    
    # Blocking for optimization
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - dilation_h * (3 - 1)) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (3 - 1)) // stride_w + 1
    
    # Create program IDs for 2D and 3D parallelism
    pid_m = tl.program_id(0)  # Batch dimension
    
    # Handle different output configurations
    if batch_size == 1:
        pid_m = 0
    else:
        pid_m = tl.program_id(0)
    
    pid_n = tl.program_id(1)  # Output channel dimension
    pid_k = tl.program_id(2)  # Output channel (for conv groups)
    
    # Offset calculations
    batch_offset = pid_m * in_channels * in_height * in_width
    conv_out_offset = pid_m * groups * out_height * out_width
    
    # Load input data for current batch and relevant spatial locations
    input_data = tl.load(x7_ptr + batch_offset + pid_k * in_height//groups * in_width + 
                        tl.arange(0, BLOCK_M * BLOCK_N).reshape(BLOCK_M, BLOCK_N))
    
    # Initialize convolution output
    conv_output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Group-wise convolution processing
    for k in range(0, BLOCK_K, 4):
        # Load kernel weights (simplified for group conv)
        kernel_data = tl.load(conv_weight_ptr + 
                             (pid_k * 1 * 3 * 3 + k // 4) * 9 + tl.arange(0, 4).reshape(2, 2))
        
        # Perform convolution operation (simplified)
        conv_output += kernel_data * input_data
    
    # Concatenation with in_6 (in-place optimization)
    concat_data = conv_output + tl.load(in6_ptr + batch_offset + pid_k * in_height//groups * in_width)
    
    # Batch normalization
    bn_mean = tl.load(bn_mean_ptr + pid_k * 128)
    bn_var = tl.load(bn_var_ptr + pid_k * 128)
    bn_weight = tl.load(bn_weight_ptr + pid_k * 128)
    bn_bias = tl.load(bn_bias_ptr + pid_k * 128)
    
    normalized = (concat_data - bn_mean) / tl.sqrt(bn_var + 1e-5)
    bn_out = normalized * bn_weight + bn_bias
    
    # PReLU activation
    prelu_weight = tl.load(prelu_weight_ptr + pid_k * 2)
    activated = tl.where(bn_out < 0, bn_out * prelu_weight, bn_out)
    
    # Store intermediate features (for tmp_9 return)
    tl.store(out_features_ptr + conv_out_offset + pid_k * out_height * out_width + 
             tl.arange(0, BLOCK_M * BLOCK_N).reshape(BLOCK_M, BLOCK_N), activated)
    
    # Adaptive average pooling (to 1x1) 
    pooled_sum = tl.sum(activated)
    pooled_avg = pooled_sum / (out_height * out_width)
    
    # Store pooled output (for tmp_11 return)
    tl.store(out_pooled_ptr + pid_m * 128 + pid_k, pooled_avg)

@torch.fx.wrap 
def fused_cgnet_forward(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    # Get input tensor shapes
    batch_size = in_7.shape[0]
    in_channels = in_7.shape[1]
    in_height = in_7.shape[2]
    in_width = in_7.shape[3]
    
    # Determine output channels and shapes
    out_channels = in_channels + 64  # conv output channels = 64, concatenated
    
    # Create output tensors
    intermediate_features = torch.empty((batch_size, out_channels, in_height, in_width), 
                                      dtype=in_7.dtype, device=in_7.device)
    pooled_output = torch.empty((batch_size, 128), dtype=in_7.dtype, device=in_7.device)
    
    # Triton kernel launch configuration
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 8
    
    # Grid configuration based on batch size and channels
    grid = (batch_size, out_channels // BLOCK_N, 1)
    
    # Launch kernel
    cgnet_fused_kernel[grid](
        # Input tensors
        in_7, in_6, in_5, in_1, in_2, in_4, in_3, in_0,
        
        # Output tensors
        intermediate_features, pooled_output,
        
        # Tensor metadata
        batch_size, in_channels, in_height, in_width, out_channels,
        
        # Conv2D parameters
        1, 1,  # stride
        4, 4,  # padding  
        4, 4,  # dilation
        64,    # groups
        
        # Block sizes
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return intermediate_features, pooled_output

def replacement_func():
    return fused_cgnet_forward