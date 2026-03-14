import torch
import triton
import triton.language as tl
import math

def pattern(x_se_74, tmp_1, tmp_0):
    """Pattern matches Conv2D followed by Sigmoid activation exactly as in model.py"""
    tmp_6 = torch.conv2d(x_se_74, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_7 = tmp_6.sigmoid()
    return tmp_7

def replacement_args(x_se_74, tmp_1, tmp_0):
    # Extract the three arguments needed for the fused kernel
    return (x_se_74, tmp_1, tmp_0)

@triton.jit
def fused_conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, C_in, C_out,
    BLOCK_SIZE_M: tl.constexpr
):
    """Simplified fused 1x1 Conv2D + Sigmoid kernel (matrix multiplication)"""
    # Each program handles one output element from one batch
    pid = tl.program_id(0)
    
    # Calculate which batch and output channel this program handles
    batch_id = pid // C_out
    c_out = pid % C_out
    
    # Check if within bounds
    if batch_id >= batch_size:
        return
    
    # Initialize accumulator for matrix multiplication
    acc = 0.0
    
    # Vector-matrix multiplication: sum over input channels
    for c_in in range(C_in):
        # Calculate linear indices for input and weights
        input_idx = batch_id * C_in + c_in
        weight_idx = c_out * C_in + c_in
        
        # Load input and weight values
        input_val = tl.load(input_ptr + input_idx, mask=c_in < C_in, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=c_in < C_in, other=0.0)
        
        # Accumulate dot product
        acc += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + c_out, mask=True, other=0.0)
    result = acc + bias_val
    
    # Apply sigmoid
    sigmoid_val = tl.sigmoid(result)
    
    # Store result
    output_idx = batch_id * C_out + c_out
    tl.store(output_ptr + output_idx, sigmoid_val, True)

@torch.fx.wrap
def fused_conv2d_sigmoid(x_se_74, tmp_1, tmp_0):
    """Simplified fused 1x1 Conv2D + Sigmoid function (matrix multiplication)"""
    # Get input dimensions for 1x1 convolution (spatial dimensions don't matter)
    batch_size = x_se_74.size(0)
    C_in = x_se_74.size(1)
    
    C_out = tmp_1.size(0)
    
    # Prepare output tensor (1x1 spatial dimensions)
    output = torch.empty(batch_size, C_out, 1, 1, device=x_se_74.device, dtype=x_se_74.dtype)
    
    # Triton launch config for matrix multiplication
    BLOCK_SIZE_M = 1  # Each program handles one output element
    
    # Calculate number of programs (one per output element)
    total_elements = batch_size * C_out
    num_programs = total_elements
    
    # Launch kernel for matrix multiplication
    fused_conv2d_sigmoid_kernel[(num_programs,)](
        x_se_74, tmp_1, tmp_0,
        output,
        batch_size, C_in, C_out,
        BLOCK_SIZE_M
    )
    
    return output

def replacement_func():
    return fused_conv2d_sigmoid