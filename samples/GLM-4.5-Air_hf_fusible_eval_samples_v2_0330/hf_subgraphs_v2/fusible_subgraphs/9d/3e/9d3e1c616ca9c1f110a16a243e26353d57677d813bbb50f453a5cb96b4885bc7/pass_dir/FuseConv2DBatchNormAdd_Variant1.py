import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input):
    """
    Pattern 1: torch.conv2d(in_6, in_4, ...) + batch_norm(..., in_0, in_1, in_3, in_2, ...) + tmp_6 += in_5
    """
    conv2d = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    batch_norm = torch.nn.functional.batch_norm(conv2d, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    result = batch_norm + residual_input
    return result

def replacement_args(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input):
    return (conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input)

@triton.jit
def fused_conv2d_bn_add_kernel(
    x_ptr,  # conv_input [N, C_in, H, W]
    w_ptr,  # conv_weight [C_out, C_in, kH, kW]
    bn_mean_ptr,  # bn_running_mean [C_out]
    bn_var_ptr,  # bn_running_var [C_out]
    bn_weight_ptr,  # bn_weight [C_out]
    bn_bias_ptr,  # bn_bias [C_out]
    residual_ptr,  # residual_input [N, C_out, H, W]
    out_ptr,  # output [N, C_out, H, W]
    N, C_in, C_out, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * H * W
    
    if mask.any():
        # Load conv output elements  
        conv_out = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Load BN parameters (using modulo to broadcast per channel)
        bn_mean = tl.load(bn_mean_ptr, mask=offsets % C_out < C_out, other=0.0).to(tl.float32)
        bn_var = tl.load(bn_var_ptr, mask=offsets % C_out < C_out, other=0.0).to(tl.float32)
        bn_weight_val = tl.load(bn_weight_ptr, mask=offsets % C_out < C_out, other=1.0).to(tl.float32)
        bn_bias_val = tl.load(bn_bias_ptr, mask=offsets % C_out < C_out, other=0.0).to(tl.float32)
        
        # Load residual input
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Apply BatchNorm per channel
        eps = 1e-05
        channel_indices = offsets % C_out
        normalized = (conv_out - bn_mean[channel_indices]) / tl.sqrt(bn_var[channel_indices] + eps)
        batch_norm_out = normalized * bn_weight_val[channel_indices] + bn_bias_val[channel_indices]
        
        # Add residual
        out_val = batch_norm_out + residual
        
        # Store result
        tl.store(out_ptr + offsets, out_val.to(tl.float32), mask=mask)

@torch.fx.wrap
def fused_conv2d_bn_add(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input):
    # Get input shapes
    N, C_in, H, W = conv_input.shape
    C_out, _, kH, kW = conv_weight.shape
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), dtype=conv_input.dtype, device=conv_input.device)
    
    # Calculate block size and grid size
    BLOCK_SIZE = 1024
    total_elements = N * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv2d_bn_add_kernel[(num_programs,)](
        conv_input,
        conv_weight,
        bn_running_mean,
        bn_running_var,
        bn_weight,
        bn_bias,
        residual_input,
        output,
        N, C_in, C_out, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv2d_bn_add