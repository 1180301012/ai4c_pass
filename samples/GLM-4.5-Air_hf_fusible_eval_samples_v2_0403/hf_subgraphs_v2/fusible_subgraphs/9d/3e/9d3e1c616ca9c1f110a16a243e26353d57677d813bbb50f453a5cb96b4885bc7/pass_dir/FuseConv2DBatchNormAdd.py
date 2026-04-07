import torch
import triton
import triton.language as tl
from typing import Tuple

def pattern(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input):
    """
    Pattern matching for Conv2D + BatchNorm + Add fusion
    
    This pattern matches the computation:
    1. conv2d = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    2. batch_norm = torch.nn.functional.batch_norm(conv2d, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    3. result = batch_norm + residual_input
    
    Note: This pattern handles both variable naming conventions across different models
    """
    conv_output = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    batch_norm_output = torch.nn.functional.batch_norm(conv_output, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    result = batch_norm_output + residual_input
    return result

def replacement_args(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input):
    """Extract arguments for the fused kernel"""
    return (conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input)

@triton.jit
def fused_conv_bn_add_kernel(
    # Input tensors
    input_ptr,           # conv_input: [N, C_in, H, W]
    weight_ptr,          # conv_weight: [C_out, C_in, K_H, K_W]
    running_mean_ptr,    # bn_running_mean: [C_out]
    running_var_ptr,     # bn_running_var: [C_out]
    weight_ptr_bn,       # bn_weight: [C_out]
    bias_ptr_bn,         # bn_bias: [C_out]
    residual_ptr,        # residual_input: [N, C_out, H, W]
    
    # Output tensor
    output_ptr,          # output: [N, C_out, H, W]
    
    # Shape information
    N: tl.constexpr,     # Batch size
    C_in: tl.constexpr,  # Input channels
    C_out: tl.constexpr, # Output channels
    H: tl.constexpr,     # Height
    W: tl.constexpr,     # Width
    K_H: tl.constexpr,   # Kernel height (1)
    K_W: tl.constexpr,   # Kernel width (1)
    
    # Strides
    input_stride_N: tl.constexpr,
    input_stride_C: tl.constexpr,
    input_stride_H: tl.constexpr,
    input_stride_W: tl.constexpr,
    
    weight_stride_C_out: tl.constexpr,
    weight_stride_C_in: tl.constexpr,
    weight_stride_K_H: tl.constexpr,
    weight_stride_K_W: tl.constexpr,
    
    residual_stride_N: tl.constexpr,
    residual_stride_C: tl.constexpr,
    residual_stride_H: tl.constexpr,
    residual_stride_W: tl.constexpr,
    
    output_stride_N: tl.constexpr,
    output_stride_C: tl.constexpr,
    output_stride_H: tl.constexpr,
    output_stride_W: tl.constexpr,
    
    # Data type
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Conv2D + BatchNorm + Add kernel"""
    pid = tl.program_id(0)
    
    # Each program handles one output channel
    c_out = pid
    
    if c_out >= C_out:
        return
    
    # Process each spatial location for this output channel
    for h_out in range(H):
        for w_out in range(W):
            # Conv2D calculation using implicit gemm for 1x1 conv
            conv_val = 0.0
            for c_in in range(C_in):
                input_val = tl.load(
                    input_ptr + h_out * input_stride_H + w_out * input_stride_W + 
                    c_in * input_stride_C,
                    mask=True,
                    other=0.0
                )
                weight_val = tl.load(
                    weight_ptr + c_out * weight_stride_C_out + 
                    c_in * weight_stride_C_in,
                    mask=True,
                    other=0.0
                )
                conv_val += input_val * weight_val
            
            # BatchNorm calculation
            running_mean_val = tl.load(running_mean_ptr + c_out)
            running_var_val = tl.load(running_var_ptr + c_out)
            weight_bn_val = tl.load(weight_ptr_bn + c_out)
            bias_bn_val = tl.load(bias_ptr_bn + c_out)
            
            # BatchNorm formula: bn_weight * (x - running_mean) / sqrt(running_var + eps) + bn_bias
            denom = tl.sqrt(running_var_val + eps)
            bn_val = weight_bn_val * (conv_val - running_mean_val) / denom + bias_bn_val
            
            # Add residual connection
            residual_val = tl.load(
                residual_ptr + h_out * residual_stride_H + w_out * residual_stride_W +
                c_out * residual_stride_C,
                mask=True,
                other=0.0
            )
            output_val = bn_val + residual_val
            
            # Store result
            tl.store(
                output_ptr + h_out * output_stride_H + w_out * output_stride_W +
                c_out * output_stride_C,
                output_val
            )

@torch.fx.wrap
def fused_conv_bn_add(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input):
    """
    Fused Conv2D + BatchNorm + Add operation using Triton kernel
    """
    # Get tensor shapes and strides
    N, C_in, H, W = conv_input.shape
    C_out = conv_weight.shape[0]
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), dtype=conv_input.dtype, device=conv_input.device)
    
    # Get strides
    input_stride = conv_input.stride()
    weight_stride = conv_weight.stride()
    residual_stride = residual_input.stride()
    output_stride = output.stride()
    
    # Kernel launch configuration
    BLOCK_SIZE = 1024  # Adjust based on performance tuning
    
    # Launch kernel with grid equal to number of output channels
    num_programs = C_out
    
    fused_conv_bn_add_kernel[(num_programs,)](
        # Input tensors
        conv_input,
        conv_weight,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        residual_input,
        
        # Output tensor
        output,
        
        # Shape information
        N, C_in, C_out, H, W, 1, 1,  # K_H=1, K_W=1 for 1x1 conv
        
        # Strides
        input_stride[0], input_stride[1], input_stride[2], input_stride[3],
        weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3],
        residual_stride[0], residual_stride[1], residual_stride[2], residual_stride[3],
        output_stride[0], output_stride[1], output_stride[2], output_stride[3],
        
        # Data type specific
        1e-05,  # epsilon for batch norm
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return fused_conv_bn_add