import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, conv_bias, scaling_factor, add_input, running_mean, running_var, weight, bias, eps, momentum):
    conv_result = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    dropout_result = torch.nn.functional.dropout(conv_result, 0.0, False, False)
    scaling_result = dropout_result * scaling_factor
    add_result = add_input + scaling_result
    batch_norm_result = torch.nn.functional.batch_norm(add_result, running_mean, running_var, weight, bias, False, momentum, eps)
    return batch_norm_result, add_result

def replacement_args(conv_input, conv_weight, conv_bias, scaling_factor, add_input, running_mean, running_var, weight, bias, eps, momentum):
    return (conv_input, conv_weight, conv_bias, scaling_factor, add_input, running_mean, running_var, weight, bias, eps, momentum)

@triton.jit
def fused_conv_batch_norm_kernel(
    input_ptr, conv_weight_ptr, conv_bias_ptr, scaling_ptr, add_input_ptr, 
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr, add_result_ptr,
    batch_size, in_channels, out_channels, height, width,
    eps: tl.constexpr, momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Map thread to spatial position and output channel
    batch_idx = pid // (height * width * ((out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE))
    remaining = pid % (height * width * ((out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE))
    h_idx = remaining // (width * ((out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE))
    w_idx = remaining // ((out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE)
    oc_group_idx = remaining % ((out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Check bounds
    if batch_idx >= batch_size or h_idx >= height or w_idx >= width:
        return
    
    # Load parameters for this batch position
    oc_start = oc_group_idx * BLOCK_SIZE
    oc_offsets = oc_start + tl.arange(0, BLOCK_SIZE)
    oc_mask = oc_offsets < out_channels
    
    # Load BN parameters
    running_mean = tl.load(running_mean_ptr + oc_offsets, mask=oc_mask, other=0.0)
    running_var = tl.load(running_var_ptr + oc_offsets, mask=oc_mask, other=1.0)
    bn_weight = tl.load(bn_weight_ptr + oc_offsets, mask=oc_mask, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
    
    # Load convolution scaling and bias
    conv_scaling = tl.load(scaling_ptr + oc_offsets, mask=oc_mask, other=1.0)
    conv_bias = tl.load(conv_bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
    
    # Process each output channel in the group
    for oc_idx in oc_offsets[oc_mask]:
        # Calculate input indices
        input_idx = ((batch_idx * height + h_idx) * width + w_idx) * in_channels
        weight_idx = oc_idx * in_channels
        
        # Load input and weights for this channel
        input_vals = tl.load(input_ptr + input_idx + tl.arange(0, in_channels), 
                           mask=input_idx + tl.arange(0, in_channels) < (batch_size * height * width * in_channels), 
                           other=0.0)
        weight_vals = tl.load(conv_weight_ptr + weight_idx + tl.arange(0, in_channels), 
                            mask=weight_idx + tl.arange(0, in_channels) < (out_channels * in_channels), 
                            other=0.0)
        
        # Compute 1x1 convolution: sum(input * weight)
        conv_result = tl.sum(input_vals * weight_vals) + conv_bias[oc_idx]
        
        # Apply scaling (layer scale)
        scaled_result = conv_result * conv_scaling[oc_idx]
        
        # Load addition input
        add_idx = ((batch_idx * height + h_idx) * width + w_idx) * out_channels + oc_idx
        add_input_val = tl.load(add_input_ptr + add_idx, 
                              mask=add_idx < (batch_size * height * width * out_channels), 
                              other=0.0)
        
        # Add input
        add_result = add_input_val + scaled_result
        
        # Store intermediate addition result
        tl.store(add_result_ptr + add_idx, add_result)
        
        # Compute batch normalization
        var_inv = tl.rsqrt(running_var[oc_idx - oc_start] + eps)
        normalized_val = (add_result - running_mean[oc_idx - oc_start]) * var_inv
        bn_result = normalized_val * bn_weight[oc_idx - oc_start] + bn_bias[oc_idx - oc_start]
        
        # Store final batch norm result
        tl.store(output_ptr + add_idx, bn_result)

@torch.fx.wrap
def fused_conv_batch_norm(conv_input, conv_weight, conv_bias, scaling_factor, add_input, 
                         running_mean, running_var, bn_weight, bn_bias, eps=1e-05, momentum=0.1):
    batch_size, in_channels, height, width = conv_input.shape
    out_channels = conv_weight.shape[0]
    
    # Output shapes
    bn_output = torch.empty((batch_size, out_channels, height, width), dtype=conv_input.dtype, device=conv_input.device)
    add_output = torch.empty((batch_size, out_channels, height, width), dtype=conv_input.dtype, device=conv_input.device)
    
    # Block size for thread scheduling
    BLOCK_SIZE = min(64, out_channels)
    
    # Calculate grid size: each thread processes one spatial position and a group of output channels
    grid_size = (batch_size * height * width * ((out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE),)
    
    fused_conv_batch_norm_kernel[grid_size](
        conv_input, conv_weight, conv_bias, scaling_factor, add_input,
        running_mean, running_var, weight, bias,
        bn_output, add_output,
        batch_size, in_channels, out_channels, height, width,
        eps=eps, momentum=momentum,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return bn_output, add_output

def replacement_func():
    return fused_conv_batch_norm