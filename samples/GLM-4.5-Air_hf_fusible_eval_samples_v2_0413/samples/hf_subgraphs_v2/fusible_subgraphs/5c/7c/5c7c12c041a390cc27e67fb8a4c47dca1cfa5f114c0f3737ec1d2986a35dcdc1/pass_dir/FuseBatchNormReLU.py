import torch
import triton
import triton.language as tl


def pattern(tmp_6, bn_mean, bn_var, bn_weight, bn_bias):
    # Match the exact computation pattern from model.py
    # batch_norm followed by relu
    # This handles both direct parameter usage and intermediate variable cases
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8


def replacement_args(tmp_6, bn_mean, bn_var, bn_weight, bn_bias):
    return (tmp_6, bn_mean, bn_var, bn_weight, bn_bias)


@triton.jit
def fused_batchnorm_relu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel index for batch norm parameters
    # For 4D tensor [N, C, H, W], each channel has H * W elements
    spatial_size_per_channel = tmp_6.shape[2] * tmp_6.shape[3] if len(tmp_6.shape) == 4 else tmp_6.shape[2]
    channel_indices = offsets // spatial_size_per_channel
    channel_indices = channel_indices % in_0.numel()  # Ensure we don't exceed parameter tensor size
    
    # Load batch norm parameters
    mean = tl.load(running_mean_ptr + channel_indices, mask=mask.any(), other=0.0)
    var = tl.load(running_var_ptr + channel_indices, mask=mask.any(), other=0.0)
    weight = tl.load(weight_ptr + channel_indices, mask=mask.any(), other=1.0)
    bias = tl.load(bias_ptr + channel_indices, mask=mask.any(), other=0.0)
    
    # Apply batch normalization
    inv_std = tl.rsqrt(var + eps)
    normalized = (input_val - mean) * inv_std * weight + bias
    
    # Apply ReLU
    output = tl.maximum(normalized, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def fused_batchnorm_relu(tmp_6, bn_mean, bn_var, bn_weight, bn_bias):
    # Get tensor properties
    input_shape = tmp_6.shape
    n_elements = tmp_6.numel()
    
    # Choose block size based on tensor characteristics
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(tmp_6)
    
    # Launch kernel
    fused_batchnorm_relu_kernel[(num_programs,)](
        input_ptr=tmp_6,
        running_mean_ptr=bn_mean,
        running_var_ptr=bn_var,
        weight_ptr=bn_weight,
        bias_ptr=bn_bias,
        output_ptr=output,
        n_elements=n_elements,
        eps=0.001,
        momentum=0.1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_batchnorm_relu