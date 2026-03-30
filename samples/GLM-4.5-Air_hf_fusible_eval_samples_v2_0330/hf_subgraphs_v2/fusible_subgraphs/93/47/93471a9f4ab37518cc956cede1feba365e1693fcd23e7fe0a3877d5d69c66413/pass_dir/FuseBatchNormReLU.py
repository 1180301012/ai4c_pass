import torch
import triton
import triton.language as tl

def pattern(conv_input, in_0, in_1, in_3, in_2):
    """Match BatchNorm + ReLU fusion pattern"""
    # BatchNorm operation
    tmp_6 = torch.nn.functional.batch_norm(conv_input, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    # ReLU operation  
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    return tmp_6, tmp_7

def replacement_args(conv_input, in_0, in_1, in_3, in_2):
    """Extract arguments for fused BatchNorm + Re kernel"""
    return conv_input, in_0, in_1, in_3, in_2

@triton.jit
def fused_batchnorm_relu_kernel(
    output_ptr, 
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    n_channels,
    height,
    width,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused BatchNorm + ReLU kernel"""
    pid = tl.program_id(0)
    n_elements = n_channels * height * width
    n_programs = tl.cdiv(n_elements, BLOCK_SIZE)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Reshape to 1D for processing
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load normalization parameters
    mean = tl.load(running_mean_ptr + (offsets // (height * width)), mask=offsets < n_channels, other=0.0)
    var = tl.load(running_var_ptr + (offsets // (height * width)), mask=offsets < n_channels, other=1.0)
    weight = tl.load(weight_ptr + (offsets // (height * width)), mask=offsets < n_channels, other=1.0)
    bias = tl.load(bias_ptr + (offsets // (height * width)), mask=offsets < n_channels, other=0.0)
    
    # Apply BatchNorm
    norm_input = (input_vals - mean) / tl.sqrt(var + eps)
    output_vals = norm_input * weight + bias
    
    # Apply ReLU
    output_vals = tl.maximum(output_vals, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, output_vals.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_batchnorm_relu(conv_input, running_mean, running_var, weight, bias):
    """Fused BatchNorm + ReLU wrapper function"""
    # Get tensor shapes
    batch, channels, height, width = conv_input.shape
    
    # Calculate total elements and grid configuration
    n_elements = channels * height * width
    BLOCK_SIZE = 1024  # Optimized block size for modern GPUs
    n_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor
    output = torch.empty_like(conv_input)
    
    # Launch Triton kernel
    fused_batchnorm_relu_kernel[(n_programs,)](
        output_ptr=output,
        input_ptr=conv_input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        n_channels=channels,
        height=height,
        width=width,
        momentum=0.1,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused function"""
    return fused_batchnorm_relu