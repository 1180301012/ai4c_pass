import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input):
    """
    Pattern matching for Conv2D + BatchNorm + Add fusion
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
    num_programs = tl.cdiv(N * H * W, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * H * W
    
    if mask.any():
        # Reshape offsets to [N, H, W]
        total_HW = H * W
        n_idx = offsets // total_HW
        hw_idx = offsets % total_HW
        h_idx = hw_idx // W
        w_idx = hw_idx % W
        
        # Calculate output spatial dimensions (assuming 1x1 conv with stride 1, padding 0)
        out_H = H
        out_W = W
        
        # Load conv output elements
        conv_out = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Load BN parameters
        bn_mean = tl.load(bn_mean_ptr + n_idx % C_out, mask=mask, other=0.0).to(tl.float32)
        bn_var = tl.load(bn_var_ptr + n_idx % C_out, mask=mask, other=0.0).to(tl.float32)
        bn_weight_val = tl.load(bn_weight_ptr + n_idx % C_out, mask=mask, other=1.0).to(tl.float32)
        bn_bias_val = tl.load(bn_bias_ptr + n_idx % C_out, mask=mask, other=0.0).to(tl.float32)
        
        # Load residual input
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Apply BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
        eps = 1e-05
        normalized = (conv_out - bn_mean) / tl.sqrt(bn_var + eps)
        batch_norm_out = normalized * bn_weight_val + bn_bias_val
        
        # Add residual
        out_val = batch_norm_out + residual
        
        # Store result
        tl.store(out_ptr + offsets, out_val.to(tl.float32), mask=mask)

@torch.fx.wrap
def fused_conv2d_bn_add(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, residual_input):
    # Get input shapes
    N, C_in, H, W = conv_input.shape
    C_out, _, kH, kW = conv_weight.shape
    
    # For 1x1 conv, output spatial dimensions are same as input
    out_H = H
    out_W = W
    
    # Create output tensor
    output = torch.empty((N, C_out, out_H, out_W), dtype=conv_input.dtype, device=conv_input.device)
    
    # Calculate block size and grid size
    BLOCK_SIZE = 1024
    total_elements = N * out_H * out_W
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