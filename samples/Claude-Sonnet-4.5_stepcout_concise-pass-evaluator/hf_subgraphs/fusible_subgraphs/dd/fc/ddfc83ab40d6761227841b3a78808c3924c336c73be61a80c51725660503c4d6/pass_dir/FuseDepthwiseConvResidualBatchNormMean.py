import torch
import triton
import triton.language as tl


def pattern(running_mean, running_var, bn_bias, bn_weight, conv_bias, conv_weight, input_tensor, residual_input):
    """
    Pattern matches: depthwise 1x1 conv + 2 residual adds + batch_norm + mean
    """
    # Depthwise 1x1 convolution
    conv_out = torch.conv2d(input_tensor, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), input_tensor.shape[1])
    
    # First residual addition
    add1 = residual_input + conv_out
    
    # Second residual addition
    add2 = add1 + input_tensor
    
    # Batch normalization (inference mode)
    bn_out = torch.nn.functional.batch_norm(add2, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    # Mean pooling over spatial dimensions
    mean_out = bn_out.mean((2, 3), keepdim=True)
    
    return (bn_out, mean_out)


def replacement_args(running_mean, running_var, bn_bias, bn_weight, conv_bias, conv_weight, input_tensor, residual_input):
    return (input_tensor, residual_input, conv_weight, conv_bias, running_mean, running_var, bn_weight, bn_bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 4}, num_warps=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def fused_kernel(
    input_ptr, residual_ptr, conv_weight_ptr, conv_bias_ptr,
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr, mean_output_ptr,
    B, C, H, W,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Fused kernel: depthwise conv (1x1) + 2 adds + batch_norm + mean pooling
    Grid: (B, C)
    Each program handles one (batch, channel) pair
    """
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Load per-channel parameters
    conv_w = tl.load(conv_weight_ptr + channel_idx)
    conv_b = tl.load(conv_bias_ptr + channel_idx)
    rm = tl.load(running_mean_ptr + channel_idx)
    rv = tl.load(running_var_ptr + channel_idx)
    bn_w = tl.load(bn_weight_ptr + channel_idx)
    bn_b = tl.load(bn_bias_ptr + channel_idx)
    
    # Compute batch norm factor: bn_w / sqrt(rv + eps)
    bn_scale = bn_w / tl.sqrt(rv + eps)
    bn_offset = bn_b - rm * bn_scale
    
    # Base offset for this (batch, channel)
    base_offset = batch_idx * C * H * W + channel_idx * H * W
    
    # Accumulator for mean
    sum_val = 0.0
    
    # Process spatial dimensions in blocks
    for h_start in range(0, H, BLOCK_H):
        for w_start in range(0, W, BLOCK_W):
            # Compute offsets for this block
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            w_offsets = w_start + tl.arange(0, BLOCK_W)
            
            # Create 2D mask
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            
            # Compute spatial indices
            h_expanded = h_offsets[:, None]
            w_expanded = w_offsets[None, :]
            offsets_2d = h_expanded * W + w_expanded
            mask_2d = h_mask[:, None] & w_mask[None, :]
            
            # Load input and residual
            input_offsets = base_offset + offsets_2d
            input_vals = tl.load(input_ptr + input_offsets, mask=mask_2d, other=0.0)
            residual_vals = tl.load(residual_ptr + input_offsets, mask=mask_2d, other=0.0)
            
            # Depthwise 1x1 conv: just scale and bias per channel
            conv_vals = input_vals * conv_w + conv_b
            
            # Two residual additions
            add1_vals = residual_vals + conv_vals
            add2_vals = add1_vals + input_vals
            
            # Batch norm
            bn_vals = add2_vals * bn_scale + bn_offset
            
            # Store output
            tl.store(output_ptr + input_offsets, bn_vals, mask=mask_2d)
            
            # Accumulate for mean
            sum_val += tl.sum(tl.where(mask_2d, bn_vals, 0.0))
    
    # Compute mean and store
    mean_val = sum_val / (H * W)
    mean_offset = batch_idx * C + channel_idx
    tl.store(mean_output_ptr + mean_offset, mean_val)


@torch.fx.wrap
def fused_depthwise_conv_bn_mean(input_tensor, residual_input, conv_weight, conv_bias, 
                                  running_mean, running_var, bn_weight, bn_bias):
    B, C, H, W = input_tensor.shape
    
    # Ensure conv_weight is squeezed to 1D
    conv_weight_1d = conv_weight.squeeze()
    
    # Allocate outputs
    output = torch.empty_like(input_tensor)
    mean_output = torch.empty((B, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with grid (B, C)
    grid = (B, C)
    
    fused_kernel[grid](
        input_tensor, residual_input, conv_weight_1d, conv_bias,
        running_mean, running_var, bn_weight, bn_bias,
        output, mean_output.view(B, C),
        B, C, H, W,
        eps=1e-05,
    )
    
    return output, mean_output


def replacement_func():
    return fused_depthwise_conv_bn_mean