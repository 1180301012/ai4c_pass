import torch
import triton
import triton.language as tl
import math

def pattern_4x4(conv_input, conv_weight, conv_bias, ln_weight, ln_bias, cls_token, input_feature):
    # Conv2D operation with 4x4 kernel (for specific graphs)
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (4, 4), (0, 0), (1, 1), 1)
    
    # Flatten to [B, C, H*W] 
    flatten_out = conv_out.flatten(2)
    
    # Transpose to [B, H*W, C]
    transpose_out = flatten_out.transpose(1, 2)
    
    # LayerNorm
    layer_norm_out = torch.nn.functional.layer_norm(transpose_out, (ln_weight.shape[0],), ln_weight, ln_bias, 1e-05)
    
    # cls_token will be handled by a separate pass
    return (conv_out, flatten_out, transpose_out, layer_norm_out)

def pattern_2x2(conv_input, conv_weight, conv_bias, ln_weight, ln_bias, cls_token, input_feature):
    # Conv2D operation with 2x2 kernel (for specific graphs)
    conv_out = torch.conv2d(conv_input, conv_weight, conv_bias, (2, 2), (0, 0), (1, 1), 1)
    
    # Flatten to [B, C, H*W] 
    flatten_out = conv_out.flatten(2)
    
    # Transpose to [B, H*W, C]
    transpose_out = flatten_out.transpose(1, 2)
    
    # LayerNorm
    layer_norm_out = torch.nn.functional.layer_norm(transpose_out, (ln_weight.shape[0],), ln_weight, ln_bias, 1e-05)
    
    # cls_token will be handled by a separate pass
    return (conv_out, flatten_out, transpose_out, layer_norm_out)

def replacement_args(conv_input, conv_weight, conv_bias, ln_weight, ln_bias, cls_token, input_feature):
    return (conv_input, conv_weight, conv_bias, ln_weight, ln_bias, cls_token, input_feature)

# Triton kernel that fuses conv2d, flatten, transpose, and layer_norm
@triton.jit
def fused_conv_norm_kernel(
    x_ptr,                      # Input tensor [B, C_in, H, W]
    weight_ptr,                 # Conv weight [C_out, C_in, K, K]
    bias_ptr,                   # Conv bias [C_out]
    ln_weight_ptr,              # Layer norm weight [C_out]
    ln_bias_ptr,                # Layer norm bias [C_out]
    out_ptr,                    # Output [B, H*W, C_out]
    B: tl.constexpr,            # Batch size
    C_in: tl.constexpr,         # Input channels
    H: tl.constexpr,            # Input height
    W: tl.constexpr,            # Input width
    C_out: tl.constexpr,        # Output channels
    K: tl.constexpr,            # Kernel size (adaptive)
    stride: tl.constexpr,       # Stride [1, 1]
    pad: tl.constexpr,          # Padding [0, 0]
    eps: tl.constexpr,          # Layer norm epsilon
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one output element [b, hw_idx, c_out]
    b = tl.program_id(0)
    hw_idx = tl.program_id(1)
    c_out = tl.program_id(2)
    
    # Calculate spatial position from flattened index
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    # Calculate input coordinates based on stride and pad
    in_h_idx = h_idx * stride[0] - pad[0]
    in_w_idx = w_idx * stride[1] - pad[1]
    
    # Initialize output
    output = tl.load(bias_ptr + c_out, other=0.0)
    
    # Simplified convolution loop
    for c in range(C_in):
        # Load weight tensor for this channel [K, K]
        weight_offset = c_out * C_in * K * K + c * K * K
        for kh in range(K):
            for kw in range(K):
                weight_val = tl.load(weight_ptr + weight_offset + kh * K + kw, other=0.0)
                
                # Calculate input coordinate with bounds check
                in_h = in_h_idx + kh
                in_w = in_w_idx + kw
                
                if 0 <= in_h < H and 0 <= in_w < W:
                    # Load input value
                    input_offset = b * C_in * H * W + c * H * W + in_h * W + in_w
                    input_val = tl.load(x_ptr + input_offset, other=0.0)
                    output += weight_val * input_val
    
    # Apply layer normalization
    mean = output  # Single element, so mean is the value itself
    var = 0.0  # Single element, variance is 0
    std = tl.sqrt(var + eps)
    
    # Load layer norm params
    weight = tl.load(ln_weight_ptr + c_out, other=1.0)
    bias = tl.load(ln_bias_ptr + c_out, other=0.0)
    
    # Apply layer norm
    normalized = (output - mean) / std * weight + bias
    
    # Store result
    out_idx = b * (H * W) * C_out + hw_idx * C_out + c_out
    tl.store(out_ptr + out_idx, normalized)

@torch.fx.wrap
def fused_conv_norm(x, weight, bias, ln_weight, ln_bias):
    B, C_in, H, W = x.shape
    C_out = ln_weight.shape[0]
    K = weight.shape[-1]  # Dynamic kernel size from weight tensor
    
    # Calculate output dimensions (with stride=1, pad=0)
    out_H = H
    out_W = W
    
    # Prepare output tensor [B, H*W, C_out]
    out_shape = (B, out_H * out_W, C_out)
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set up launch parameters
    BLOCK_SIZE_C = 128  # Tunable parameter
    
    # Calculate grid dimensions
    grid = (
        B,
        out_H * out_W,
        C_out
    )
    
    # Launch kernel
    fused_conv_norm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        ln_weight_ptr=ln_weight,
        ln_bias_ptr=ln_bias,
        out_ptr=output,
        B=B,
        C_in=C_in,
        H=H,
        W=W,
        C_out=C_out,
        K=K,
        stride=(1, 1),
        pad=(0, 0),
        eps=1e-05,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return output

def replacement_func():
    return fused_conv_norm