import torch
import triton
import triton.language as tl
from typing import Tuple

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Pattern matches the computation:
    tmp_6 = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), channels)
    tmp_7 = other_input + tmp_6  
    tmp_8 = tmp_7 + input
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_10 = tmp_9.mean((2, 3), keepdim=True)
    Returns (tmp_9, tmp_10) to match observable outputs
    """
    tmp_0 = in_0
    tmp_1 = in_1  
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_4
    tmp_5 = in_5
    
    # Match conv2d - note it uses positional args, not keyword
    tmp_6 = torch.conv2d(in_6, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 192)  # channels will be extracted from pattern
    
    # The additions pattern - adapts based on conv input
    tmp_7 = in_7 + tmp_6
    tmp_8 = tmp_7 + in_6
    
    # Batch normalization - uses positional args
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_10 = tmp_9.mean((2, 3), keepdim=True)
    
    return tmp_9, tmp_10

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)

@triton.jit
def fused_conv_add_bn_kernel(
    # Input pointers
    input_ptr,           # Pointer to input tensor used in conv
    other_input_ptr,     # Pointer to the other input tensor
    running_mean_ptr,    # Batch norm running mean
    running_var_ptr,     # Batch norm running var
    weight_ptr,          # Batch norm weight (scale)
    bias_ptr,            # Batch norm bias
    conv_weight_ptr,     # Conv2d weight
    conv_bias_ptr,       # Conv2d bias
    
    # Output pointers  
    bn_output_ptr,       # Batch norm output
    mean_output_ptr,     # Mean output
    
    # Tensor shapes
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    
    # Strides
    input_stride_batch: tl.constexpr,
    input_stride_channels: tl.constexpr,
    input_stride_height: tl.constexpr,
    input_stride_width: tl.constexpr,
    
    other_input_stride_batch: tl.constexpr,
    other_input_stride_channels: tl.constexpr,
    other_input_stride_height: tl.constexpr,
    other_input_stride_width: tl.constexpr,
    
    # Conv parameters (fixed for 1x1 conv)
    conv_stride_h: tl.constexpr,
    conv_stride_w: tl.constexpr,
    conv_padding_h: tl.constexpr,
    conv_padding_w: tl.constexpr,
    conv_dilation_h: tl.constexpr,
    conv_dilation_w: tl.constexpr,
    
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # channels per program
    BLOCK_SIZE_N: tl.constexpr,  # spatial elements per program
):
    """Fused kernel: conv2d + add + add + batch_norm + mean reduction"""
    
    # Program identifiers
    m_id = tl.program_id(0)  # channel group
    n_id = tl.program_id(1)  # spatial position group
    
    # Range for this program
    m_start = m_id * BLOCK_SIZE_M
    n_start = n_id * BLOCK_SIZE_N
    m_end = min(m_start + BLOCK_SIZE_M, channels)
    n_end = min(n_start + BLOCK_SIZE_N, height * width)
    
    # Process one channel group
    for m in range(m_start, m_end):
        conv_weight = tl.load(conv_weight_ptr + m * conv_stride_channels)
        conv_bias = tl.load(conv_bias_ptr + m)
        bn_weight = tl.load(weight_ptr + m)
        bn_bias = tl.load(bias_ptr + m)
        running_mean = tl.load(running_mean_ptr + m)
        running_var = tl.load(running_var_ptr + m)
        
        # Compute spatially-invariant quantities for this channel
        conv_bn_weight = conv_weight * bn_weight
        conv_bn_bias = conv_bias * bn_weight + bn_bias - running_mean * bn_weight
        
        variance_sqrt = tl.sqrt(running_var + 1e-05)
        
        # Accumulators for mean computation
        spatial_sum = 0.0
        
        # Process spatial positions
        for n_idx in range(n_start, n_end):
            # Convert linear index to 2D spatial position
            h = n_idx // width
            w = n_idx % width
            
            # Compute input offsets
            input_offset = (0 * input_stride_batch + 
                          m * input_stride_channels + 
                          h * input_stride_height + 
                          w * input_stride_width)
            other_input_offset = (0 * other_input_stride_batch + 
                                m * other_input_stride_channels + 
                                h * other_input_stride_height + 
                                w * other_input_stride_width)
            
            # Load input values
            input_val = tl.load(input_ptr + input_offset)
            other_input_val = tl.load(other_input_ptr + other_input_offset)
            
            # Conv operation (1x1 pointwise)
            conv_result = conv_weight * input_val + conv_bias
            
            # Add operations: conv_result + other_input + input_val
            fused_add = conv_result + other_input_val + input_val
            
            # Batch normalization: (x - mean) / sqrt(var) * weight + bias
            normalized = (fused_add - running_mean) / variance_sqrt
            bn_result = normalized * bn_weight + bn_bias
            
            # Store batch norm output
            bn_output_offset = (0 * input_stride_batch + 
                              m * input_stride_channels + 
                              h * input_stride_height + 
                              w * input_stride_width)
            tl.store(bn_output_ptr + bn_output_offset, bn_result)
            
            # Accumulate for mean computation
            spatial_sum += bn_result
        
        # Store mean for this channel
        mean_val = spatial_sum / (height * width)
        mean_offset = m * channels  # mean has shape [channels, 1, 1]
        tl.store(mean_output_ptr + mean_offset, mean_val)

@torch.fx.wrap
def fused_conv_add_bn(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Wrapper function to launch the fused kernel"""
    # Get tensor properties
    batch_size, channels, height, width = in_6.shape
    
    # Create output tensors
    bn_output = torch.empty_like(in_6)
    mean_output = torch.empty((channels, 1, 1), dtype=in_6.dtype, device=in_6.device)
    
    # Get tensor strides
    input_stride_batch = in_6.stride(0)
    input_stride_channels = in_6.stride(1)
    input_stride_height = in_6.stride(2)
    input_stride_width = in_6.stride(3)
    
    other_input_stride_batch = in_7.stride(0)
    other_input_stride_channels = in_7.stride(1)
    other_input_stride_height = in_7.stride(2)
    other_input_stride_width = in_7.stride(3)
    
    # Launch kernel with autotuned block sizes
    BLOCK_SIZE_M = 64  # channels per block
    BLOCK_SIZE_N = 1024  # spatial elements per block
    
    grid = (
        (channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,  # channel blocks
        (height * width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,  # spatial blocks
    )
    
    fused_conv_add_bn_kernel[grid](
        # Input pointers
        input_ptr=in_6,
        other_input_ptr=in_7,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        conv_weight_ptr=in_5,
        conv_bias_ptr=in_4,
        
        # Output pointers
        bn_output_ptr=bn_output,
        mean_output_ptr=mean_output,
        
        # Tensor shapes
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        
        # Strides
        input_stride_batch=input_stride_batch,
        input_stride_channels=input_stride_channels,
        input_stride_height=input_stride_height,
        input_stride_width=input_stride_width,
        
        other_input_stride_batch=other_input_stride_batch,
        other_input_stride_channels=other_input_stride_channels,
        other_input_stride_height=other_input_stride_height,
        other_input_stride_width=other_input_stride_width,
        
        # Conv parameters
        conv_stride_h=1,
        conv_stride_w=1,
        conv_padding_h=0,
        conv_padding_w=0,
        conv_dilation_h=1,
        conv_dilation_w=1,
        
        # Block sizes
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return bn_output, mean_output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_add_bn