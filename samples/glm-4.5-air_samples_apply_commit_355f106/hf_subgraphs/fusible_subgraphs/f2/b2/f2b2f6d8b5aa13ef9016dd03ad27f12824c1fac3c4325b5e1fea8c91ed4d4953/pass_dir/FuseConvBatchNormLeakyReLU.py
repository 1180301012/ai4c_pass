import torch
import triton
import triton.language as tl

# Pattern matching function - matches Conv2D + BatchNorm + LeakyReLU sequence
def pattern(x, conv_weight, running_mean, running_var, bn_bias, bn_weight):
    """
    Pattern matching function for fused Conv2D + BatchNorm + LeakyReLU
    """
    # Conv2D operation with bias=None
    conv_out = torch.conv2d(x, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    
    # BatchNorm operation (training=False, momentum=0.1, eps=1e-05)
    bn_out = torch.nn.functional.batch_norm(conv_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    # LeakyReLU operation (inplace=True, negative_slope=0.01)
    relu_out = torch.nn.functional.leaky_relu(bn_out, 0.01, True)
    
    # Return only the final output to match the original function
    return relu_out,

# Argument extraction function
def replacement_args(x, conv_weight, running_mean, running_var, bn_bias, bn_weight):
    """
    Extract arguments for the fused kernel
    """
    return (x, conv_weight, running_mean, running_var, bn_bias, bn_weight)

# Optimized fused kernel
@triton.jit
def fused_conv_bn_relu_kernel(
    x_ptr,
    conv_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_bias_ptr,
    bn_weight_ptr,
    out_ptr,
    N,      # batch size
    C_out,  # output channels
    H_out,  # output height
    W_out,  # output width
    C_in,   # input channels
    KH,     # kernel height
    KW,     # kernel width,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Fused Conv2D + BatchNorm + LeakyReLU kernel
    """
    # Get program ID
    pid = tl.program_id(0)
    M = N * C_out * H_out * W_out
    
    # Compute output element position
    row = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = row < M
    
    # Flatten row to N, C_out, H_out, W_out coordinates
    row_fused = row
    n = row_fused // (C_out * H_out * W_out)
    row_fused = row_fused % (C_out * H_out * W_out)
    c_out = row_fused // (H_out * W_out)
    row_fused = row_fused % (H_out * W_out)
    h_out = row_fused // W_out
    w_out = row_fused % W_out
    
    # Compute spatial positions for convolution
    h_in = h_out * 1 - 1  # stride=1, padding=1
    w_in = w_out * 1 - 1  # stride=1, padding=1
    
    if h_in < 0 or h_in >= H_out or w_in < 0 or w_in >= W_out:
        # Out of bounds, return zero
        tl.store(out_ptr + row, 0.0, mask=mask)
        return
    
    # Load running mean and var (broadcast to batch)
    running_mean = tl.load(running_mean_ptr + c_out)
    running_var = tl.load(running_var_ptr + c_out)
    bn_weight = tl.load(bn_weight_ptr + c_out)
    bn_bias = tl.load(bn_bias_ptr + c_out)
    
    # Initialize output accumulator for this output pixel
    acc = 0.0
    
    # Convolution computation
    for kh in range(KH):
        for kw in range(KW):
            for ci in range(C_in):
                # Calculate input spatial position
                h_pos = h_in + kh
                w_pos = w_in + kw
                
                if h_pos >= 0 and h_pos < H_out and w_pos >= 0 and w_pos < W_out:
                    # Input element index: n, ci, h_pos, w_pos
                    input_idx = n * C_in * H_out * W_out + ci * H_out * W_out + h_pos * W_out + w_pos
                    
                    # Weight element index: c_out, ci, kh, kw  
                    weight_idx = c_out * C_in * KH * KW + ci * KH * KW + kh * KW + kw
                    
                    # Load input and weight
                    x_val = tl.load(x_ptr + input_idx)
                    weight_val = tl.load(conv_weight_ptr + weight_idx)
                    
                    # Multiply and accumulate
                    acc += x_val * weight_val
    
    # Apply BatchNorm
    running_var_eps = running_var + 1e-05  # eps = 1e-05
    inv_std = 1.0 / tl.sqrt(running_var_eps)
    bn_val = acc * inv_std * bn_weight + bn_bias
    
    # Apply LeakyReLU
    if bn_val > 0:
        out_val = bn_val
    else:
        out_val = bn_val * 0.01  # negative_slope = 0.01
    
    # Store result
    tl.store(out_ptr + row, out_val, mask=mask)

@torch.fx.wrap
def fused_conv_bn_relu_wrapper(x, conv_weight, running_mean, running_var, bn_bias, bn_weight):
    """
    Wrapper function to launch the fused kernel
    """
    # Get input and output shapes
    N, C_in, H_in, W_in = x.shape
    K, C_in_conv, KH, KW = conv_weight.shape
    C_out = K  # output channels = number of conv filters
    
    # For stride=1, padding=1, output spatial size = input spatial size
    H_out, W_out = H_in, W_in
    
    # Create output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Block size for tiling
    BLOCK_SIZE_M = 1024
    
    # Calculate number of programs needed
    total_elements = N * C_out * H_out * W_out
    num_programs = triton.cdiv(total_elements, BLOCK_SIZE_M)
    
    # Launch kernel
    fused_conv_bn_relu_kernel[(num_programs,)](
        x_ptr=x,
        conv_weight_ptr=conv_weight,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_bias_ptr=bn_bias,
        bn_weight_ptr=bn_weight,
        out_ptr=out,
        N=N,
        C_out=C_out,
        H_out=H_out,
        W_out=W_out,
        C_in=C_in,
        KH=KH,
        KW=KW,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    """
    Returns the function reference for the fused operation
    """
    return fused_conv_bn_relu_wrapper