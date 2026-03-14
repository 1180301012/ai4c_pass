import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(running_mean, running_var, bn_bias, bn_weight, conv_weight, residual, input_tensor):
    """
    Match the pattern: Conv2D (1x1) -> BatchNorm -> Add
    """
    # Conv2D with 1x1 kernel
    conv_out = torch.conv2d(input_tensor, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    
    # BatchNorm
    bn_out = torch.nn.functional.batch_norm(
        conv_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05
    )
    
    # Add residual
    bn_out += residual
    result = bn_out
    
    return result


def replacement_args(running_mean, running_var, bn_bias, bn_weight, conv_weight, residual, input_tensor):
    return (running_mean, running_var, bn_bias, bn_weight, conv_weight, residual, input_tensor)


@triton.jit
def fused_conv_bn_add_kernel(
    input_ptr,
    conv_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    residual_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    """
    Fused kernel for Conv2D (1x1) + BatchNorm + Add
    Each program processes one spatial location (b, h, w) and a block of output channels
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate batch, height, width indices
    spatial_size = height * width
    total_spatial = batch_size * spatial_size
    num_oc_blocks = tl.cdiv(out_channels, BLOCK_SIZE_OC)
    
    spatial_idx = pid // num_oc_blocks
    oc_block_idx = pid % num_oc_blocks
    
    if spatial_idx >= total_spatial:
        return
    
    b = spatial_idx // spatial_size
    hw = spatial_idx % spatial_size
    h = hw // width
    w = hw % width
    
    # Output channel range for this block
    oc_start = oc_block_idx * BLOCK_SIZE_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < out_channels
    
    # Initialize accumulator for convolution
    acc = tl.zeros([BLOCK_SIZE_OC], dtype=tl.float32)
    
    # Perform 1x1 convolution: sum over input channels
    for ic_start in range(0, in_channels, BLOCK_SIZE_IC):
        ic_offsets = ic_start + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < in_channels
        
        # Load input: shape [BLOCK_SIZE_IC]
        input_idx = b * in_channels * height * width + ic_offsets * height * width + h * width + w
        input_vals = tl.load(input_ptr + input_idx, mask=ic_mask, other=0.0)
        
        # Load conv weights: shape [BLOCK_SIZE_OC, BLOCK_SIZE_IC]
        # Weight layout: [out_channels, in_channels, 1, 1]
        for ic_i in range(BLOCK_SIZE_IC):
            ic = ic_start + ic_i
            if ic < in_channels:
                weight_idx = oc_offsets * in_channels + ic
                weights = tl.load(conv_weight_ptr + weight_idx, mask=oc_mask, other=0.0)
                acc += weights * input_vals[ic_i]
    
    # Apply BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    mean = tl.load(running_mean_ptr + oc_offsets, mask=oc_mask, other=0.0)
    var = tl.load(running_var_ptr + oc_offsets, mask=oc_mask, other=0.0)
    bn_w = tl.load(bn_weight_ptr + oc_offsets, mask=oc_mask, other=0.0)
    bn_b = tl.load(bn_bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
    
    normalized = (acc - mean) / tl.sqrt(var + eps)
    bn_out = normalized * bn_w + bn_b
    
    # Add residual
    residual_idx = b * out_channels * height * width + oc_offsets * height * width + h * width + w
    residual_vals = tl.load(residual_ptr + residual_idx, mask=oc_mask, other=0.0)
    output = bn_out + residual_vals
    
    # Store output
    output_idx = b * out_channels * height * width + oc_offsets * height * width + h * width + w
    tl.store(output_ptr + output_idx, output, mask=oc_mask)


@torch.fx.wrap
def fused_conv_bn_add(running_mean, running_var, bn_bias, bn_weight, conv_weight, residual, input_tensor):
    """
    Wrapper function for the fused kernel
    """
    # Get dimensions
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = conv_weight.shape[0]
    
    # Prepare output
    output = torch.empty(batch_size, out_channels, height, width, 
                        device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Grid: one program per spatial location and output channel block
    BLOCK_SIZE_OC = 64
    BLOCK_SIZE_IC = 32
    spatial_size = batch_size * height * width
    num_oc_blocks = triton.cdiv(out_channels, BLOCK_SIZE_OC)
    grid = (spatial_size * num_oc_blocks,)
    
    eps = 1e-05
    
    fused_conv_bn_add_kernel[grid](
        input_tensor,
        conv_weight,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        residual,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        eps,
        BLOCK_SIZE_OC,
        BLOCK_SIZE_IC,
    )
    
    return output


def replacement_func():
    return fused_conv_bn_add