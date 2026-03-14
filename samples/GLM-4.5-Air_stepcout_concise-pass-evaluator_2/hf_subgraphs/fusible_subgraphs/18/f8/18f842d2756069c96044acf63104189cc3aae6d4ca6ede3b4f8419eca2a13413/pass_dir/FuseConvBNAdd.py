import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    """Match Conv2D + BatchNorm + Addition pattern"""
    tmp_5 = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = tmp_6 + in_5
    return tmp_7

# Argument extraction function
def replacement_args(in_6, in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_6, in_4, in_0, in_1, in_3, in_2, in_5)

# Triton kernel for fused Conv2D + BatchNorm + Add
@triton.jit
def fused_conv_bn_add_kernel(
    x_ptr,  # input activation [N, C_in, H, W]
    weight_ptr,  # conv weight [C_out, C_in, 1, 1]
    bn_mean_ptr,  # batch norm running mean [C_out]
    bn_var_ptr,  # batch norm running var [C_out]
    bn_weight_ptr,  # batch norm weight [C_out]
    bn_bias_ptr,  # batch norm bias [C_out]
    residual_ptr,  # residual input [N, C_out, H, W]
    out_ptr,  # output [N, C_out, H, W]
    N, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Grid mapping: each CTA handles a slice of channels
    c_out = tl.program_id(0)
    n_start = tl.program_id(1) * BLOCK_SIZE_N
    h_start = tl.program_id(2) * BLOCK_SIZE_HW
    w_start = tl.program_id(3) * BLOCK_SIZE_HW
    
    # Load batch norm parameters for this output channel
    mean = tl.load(bn_mean_ptr + c_out)
    var = tl.load(bn_var_ptr + c_out)
    weight = tl.load(bn_weight_ptr + c_out)
    bias = tl.load(bn_bias_ptr + c_out)
    
    # Precompute batch norm scale and shift
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(var + eps)
    bn_scale = weight * inv_std
    bn_bias_corrected = bias - mean * bn_scale
    
    # Process batch and spatial tiles
    for n in range(n_start, min(n_start + BLOCK_SIZE_N, N)):
        for h in range(h_start, min(h_start + BLOCK_SIZE_HW, H)):
            for w in range(w_start, min(w_start + BLOCK_SIZE_HW, W)):
                # Load input activation (1x1 conv, so spatial locations same)
                in_val = tl.load(x_ptr + n * H * W + h * W + w)
                
                # Load conv weight (1x1, so spatial index doesn't matter)
                weight_val = tl.load(weight_ptr + c_out)
                
                # Compute 1x1 convolution
                conv_val = in_val * weight_val
                
                # Apply batch normalization
                bn_val = conv_val * bn_scale + bn_bias_corrected
                
                # Load residual and add
                residual_val = tl.load(residual_ptr + (n * H * W + h * W + w) * 1024 + c_out)
                out_val = bn_val + residual_val
                
                # Store result
                out_idx = (n * H * W + h * W + w) * 1024 + c_out
                tl.store(out_ptr + out_idx, out_val)

# Kernel wrapper
@torch.fx.wrap
def fused_conv_bn_add(x, weight, bn_mean, bn_var, bn_weight, bn_bias, residual):
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Create output tensor
    out = torch.empty((N, C_out, H, W), dtype=torch.float32, device=x.device)
    
    # Triton grid setup
    BLOCK_SIZE_N = 1  # Process one sample at a time for simplicity
    BLOCK_SIZE_HW = 64  # 8x8 tile size
    BLOCK_SIZE_C = 32  # Process 32 channels at a time
    
    grid = (
        (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C,  # Channel dimension
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,     # Batch dimension  
        (H + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW,    # Height dimension
        (W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW,    # Width dimension
    )
    
    fused_conv_bn_add_kernel[grid](
        x,
        weight,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        residual,
        out,
        N, H, W,
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
        BLOCK_SIZE_HW
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_conv_bn_add