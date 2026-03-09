import torch
import triton
import triton.language as tl

# Pattern for Conv2D + BatchNorm + ReLU fusion
def pattern(conv_input, conv_weight, batch_norm_mean, batch_var, batch_weight, batch_bias):
    # Match the sequence of operations
    conv_output = torch.conv2d(input=conv_input, weight=conv_weight, groups=512)
    
    # BatchNorm parameters (running_mean, running_var, weight, bias, momentum, eps)
    batch_norm_output = torch.nn.functional.batch_norm(
        conv_output, 
        batch_norm_mean, 
        batch_var, 
        batch_weight, 
        batch_bias, 
        False,  # training=False
        0.1,     # momentum
        1e-05    # epsilon
    )
    
    # ReLU
    relu_output = torch.nn.functional.relu(batch_norm_output, inplace=False)
    
    return relu_output

def replacement_args(conv_input, conv_weight, batch_norm_mean, batch_var, batch_weight, batch_bias):
    return (conv_input, conv_weight, batch_norm_mean, batch_var, batch_weight, batch_bias)

# Optimized fused kernel: Conv2D + BatchNorm + ReLU
@triton.jit
def fused_conv_batch_norm_relu_kernel(
    x_ptr, 
    weight_ptr, 
    mean_ptr,
    var_ptr,
    gamma_ptr,  # batch weight
    beta_ptr,   # batch bias
    out_ptr,
    B, C, H_in, W_in, H_out, W_out,
    KH, KW,
    C_norm,    # number of channels for normalization
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    # Get program ID
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Compute output block coordinates
    h_out_base = pid_h * BLOCK_SIZE_H
    w_out_base = pid_w * BLOCK_SIZE_W
    c_block = tl.program_id(2)
    c_start = c_block * BLOCK_SIZE_C
    c_end = min(c_start + BLOCK_SIZE_C, C)
    
    # Compute input coordinates with padding and stride
    h_in_base = h_out_base * stride - padding
    w_in_base = w_out_base * stride - padding
    
    # Initialize output accumulator
    out = tl.zeros([BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W], dtype=tl.float32)
    
    # Load normalization parameters for this channel block
    if c_start < C:
        # Load batch norm parameters
        mean = tl.load(mean_ptr + c_start, eviction_policy='evict_last')
        var = tl.load(var_ptr + c_start, eviction_policy='evict_last')
        gamma = tl.load(gamma_ptr + c_start, eviction_policy='evict_last')
        beta = tl.load(beta_ptr + c_start, eviction_policy='evict_last')
        
        # Compute inverse standard deviation
        inv_std = 1.0 / tl.sqrt(var + 1e-05)
        
        # Compute normalization scaling factors
        norm_scale = gamma * inv_std
        norm_bias = beta - mean * norm_scale
    
    # Compute convolution for this output block
    for c in range(c_start, c_end):
        if c < C:
            for kh in range(KH):
                for kw in range(KW):
                    # Compute input coordinates for this kernel position
                    h_in = h_in_base + kh * dilation
                    w_in = w_in_base + kw * dilation
                    
                    # Check bounds
                    h_in_valid = (h_in >= 0) & (h_in < H_in)
                    w_in_valid = (w_in >= 0) & (w_in < W_in)
                    if h_in_valid and w_in_valid:
                        # Load input value
                        x_offset = (0, c, h_in, w_in)
                        x_val = tl.load(x_ptr + x_offset, eviction_policy='evict_last')
                        
                        # Load weight value (depthwise convolution)
                        w_offset = (c % C, 0, kh, kw)
                        w_val = tl.load(weight_ptr + w_offset, eviction_policy='evict_last')
                        
                        # Convolution: multiply and accumulate
                        conv_val = x_val * w_val
    
                        # Apply BatchNorm
                        if c_start < C:
                            norm_val = conv_val * norm_scale + norm_bias
                        else:
                            norm_val = conv_val
                        
                        # Apply ReLU
                        relu_val = tl.maximum(norm_val, 0.0)
                        
                        # Store to accumulator
                        out_h = h_out_base % BLOCK_SIZE_H
                        out_w = w_out_base % BLOCK_SIZE_W
                        out_c = c - c_start
                        out[out_c, out_h, out_w] += relu_val
    
    # Store output block
    if c_start < C:
        for h_out in range(BLOCK_SIZE_H):
            for w_out in range(BLOCK_SIZE_W):
                h_out_valid = (h_out_base + h_out < H_out)
                w_out_valid = (w_out_base + w_out < W_out)
                if h_out_valid and w_out_valid:
                    for c_out in range(BLOCK_SIZE_C):
                        c_idx = c_start + c_out
                        if c_idx < C:
                            out_offset = (0, c_idx, h_out_base + h_out, w_out_base + w_out)
                            tl.store(out_ptr + out_offset, out[c_out, h_out % BLOCK_SIZE_H, w_out % BLOCK_SIZE_W])

@torch.fx.wrap
def optimized_fused_conv_batch_norm_relu(conv_input, conv_weight, batch_norm_mean, batch_var, batch_weight, batch_bias):
    # Get input dimensions
    B, C, H_in, W_in = conv_input.shape
    KH, KW = conv_weight.shape[2], conv_weight.shape[3]
    
    # Calculate output dimensions (assuming stride=1, padding=0 for simplicity)
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    
    # Create output tensor
    output = torch.empty((B, C, H_out, W_out), dtype=torch.float32, device=conv_input.device)
    
    # Choose block sizes for better occupancy
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    BLOCK_SIZE_C = 32  # Process multiple channels per program
    
    # Calculate grid dimensions
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    fused_conv_batch_norm_relu_kernel[(grid_h, grid_w, grid_c)](
        conv_input, conv_weight, batch_norm_mean, batch_var, batch_weight, batch_bias,
        output,
        B, C, H_in, W_in, H_out, W_out,
        KH, KW, C,
        stride=1, padding=0, dilation=1,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return output

def replacement_func():
    return optimized_fused_conv_batch_norm_relu