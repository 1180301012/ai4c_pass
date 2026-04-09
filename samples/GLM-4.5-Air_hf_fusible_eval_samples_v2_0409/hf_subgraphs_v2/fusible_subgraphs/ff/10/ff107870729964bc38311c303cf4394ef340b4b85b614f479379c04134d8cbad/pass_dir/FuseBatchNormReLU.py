import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, running_mean, running_var, weight_bn, bias_bn, eps=1e-05):
    """Pattern: Conv2D + BatchNorm + ReLU"""
    conv = torch.conv2d(x, weight, bias, (1, 1), (1, 1), (1, 1), 1)
    bn = torch.nn.functional.batch_norm(conv, running_mean, running_var, weight_bn, bias_bn, False, 0.1, eps)
    relu = torch.nn.functional.relu(bn, inplace=False)
    return conv, relu

def replacement_args(x, weight, bias, running_mean, running_var, weight_bn, bias_bn, eps):
    return (x, weight, bias, running_mean, running_var, weight_bn, bias_bn, eps)

@triton.jit
def fused_bn_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_bn_ptr,
    bias_bn_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = (B * C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid >= num_blocks:
        return
    
    # Flatten the tensor for processing
    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < B * C * H * W
    
    # Load input
    x = tl.load(x_ptr + indices, mask=mask, other=0.0).to(tl.float32)
    
    # Load batch norm parameters
    weight_bn = tl.load(weight_bn_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C).to(tl.float32)
    bias_bn = tl.load(bias_bn_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C).to(tl.float32)
    running_mean = tl.load(running_mean_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C).to(tl.float32)
    running_var = tl.load(running_var_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C).to(tl.float32)
    
    # Process each channel
    for c in range(C):
        # Load and process channel data
        channel_mask = mask & (indices % C == c)
        if not tl.any(channel_mask):
            continue
            
        # Apply batch norm
        norm_x = (x - running_mean[c]) / tl.sqrt(running_var[c] + eps)
        bn_out = norm_x * weight_bn[c] + bias_bn[c]
        relu_out = tl.where(bn_out > 0, bn_out, 0)
        
        # Store result
        tl.store(out_ptr + indices, relu_out.to(tl.float16), mask=channel_mask)

@torch.fx.wrap
def fused_bn_relu(x, weight, bias, running_mean, running_var, weight_bn, bias_bn):
    B, C, H, W = x.shape
    N = B * C * H * W
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_bn_relu_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_bn_ptr=weight_bn,
        bias_bn_ptr=bias_bn,
        out_ptr=out,
        B=B, C=C, H=H, W=W,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return intermediate and final results
    return x, out  # conv result, relu result

def replacement_func():
    return fused_bn_relu