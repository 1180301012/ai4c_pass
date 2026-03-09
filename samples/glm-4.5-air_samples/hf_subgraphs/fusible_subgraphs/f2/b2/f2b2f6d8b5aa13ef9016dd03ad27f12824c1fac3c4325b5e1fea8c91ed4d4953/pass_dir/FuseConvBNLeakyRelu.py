import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias):
    conv_output = torch.conv2d(conv_input, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    bn_output = torch.nn.functional.batch_norm(conv_output, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    relu_output = torch.nn.functional.leaky_relu(bn_output, 0.01, True)
    return relu_output

def replacement_args(conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias):
    return (conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias)

@triton.jit
def conv_bn_relu_kernel(
    input_ptr, weight_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    output_ptr,
    b_idx: tl.constexpr,
    C_in, H_in, W_in,
    C_out, K_H: tl.constexpr, K_W: tl.constexpr,
    H_out, W_out,
    stride_H, stride_W, pad_H, pad_W,
    eps,
    negative_slope,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Get output coordinates
    c_offset = pid_c * BLOCK_SIZE_C
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    
    c = c_offset + tl.arange(0, BLOCK_SIZE_C)
    h = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w = w_offset + tl.arange(0, BLOCK_SIZE_W)
    
    # Clamp to valid range
    c = tl.minimum(c, C_out - 1)
    h = tl.minimum(h, H_out - 1)
    w = tl.minimum(w, W_out - 1)
    
    # Initialize accumulator
    acc = tl.zeros([c.shape[0]], dtype=tl.float32)
    
    # Convolution computation
    for ci in range(C_in):
        # Get weight for this input channel
        weight_ptr_base = weight_ptr + (c[:, None, None] * C_in + ci) * K_H * K_W
        weight_ptrs = weight_ptr_base + tl.arange(0, K_H)[:, None] * K_W + tl.arange(0, K_W)
        weight_vals = tl.load(weight_ptrs, mask=weight_ptrs < C_out * C_in * K_H * K_W, other=0.0).to(tl.float32)
        
        # Calculate input spatial coordinates
        h_in = h[:, None, None] * stride_H - pad_H + tl.arange(0, K_H)[None, :, None]
        w_in = w[:, None, None] * stride_W - pad_W + tl.arange(0, K_W)[None, None, :]
        
        # Check bounds and create mask
        h_valid = (h_in >= 0) & (h_in < H_in)
        w_valid = (w_in >= 0) & (w_in < W_in)
        mask = h_valid & w_valid
        
        # Load input data for specific batch element
        input_ptr_base = input_ptr + (b_idx * C_in * H_in * W_in + ci * H_in * W_in)
        input_ptrs = input_ptr_base + h_in * W_in + w_in
        input_vals = tl.load(input_ptrs, mask=mask[:, :, :], other=0.0).to(tl.float32)
        
        # Compute partial sum
        input_weighted = input_vals * weight_vals
        acc += tl.sum(input_weighted, axis=(1, 2))
    
    # Load batch norm parameters for this channel group
    mean = tl.load(mean_ptr + c, mask=c < C_out, other=0.0)
    var = tl.load(var_ptr + c, mask=c < C_out, other=1.0)
    gamma = tl.load(gamma_ptr + c, mask=c < C_out, other=1.0)
    beta = tl.load(beta_ptr + c, mask=c < C_out, other=0.0)
    
    # Apply batch normalization
    var_rsqrt = 1.0 / tl.sqrt(var + eps)
    output_norm = (acc - mean) * var_rsqrt * gamma + beta
    
    # Apply LeakyReLU
    output_relu = tl.where(output_norm > 0, output_norm, negative_slope * output_norm)
    
    # Store results for specific batch element
    output_ptr_base = output_ptr + (b_idx * C_out * H_out * W_out + c * H_out * W_out + h * W_out + w)
    tl.store(output_ptr_base, output_relu, mask=(c < C_out)[:, None, None])

@torch.fx.wrap
def conv_bn_relu_fusion(conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias):
    # Get input dimensions
    N, C_in, H_in, W_in = conv_input.shape
    C_out, C_in_w, K_H, K_W = conv_weight.shape
    
    # Calculate output spatial dimensions (with stride=1, padding=1, dilation=1)
    H_out = ((H_in + 2 * 1 - K_H) // 1) + 1
    W_out = ((W_in + 2 * 1 - K_W) // 1) + 1
    
    # Create output tensor
    output = torch.empty((N, C_out, H_out, W_out), dtype=torch.float32, device=conv_input.device)
    
    # Configure block sizes
    BLOCK_SIZE_C = min(32, C_out)  # Adjust for channel size
    BLOCK_SIZE_H = min(16, H_out)  # Adjust for height size
    BLOCK_SIZE_W = min(16, W_out)  # Adjust for width size
    
    # Calculate grid dimensions
    grid_c = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel for each sample in batch
    for b in range(N):
        conv_bn_relu_kernel[grid_c, grid_h, grid_w](
            conv_input[b], conv_weight, running_mean, running_var, bn_weight, bn_bias,
            output[b],
            b,  # batch index as constexpr
            C_in, H_in, W_in,
            C_out, K_H, K_W,
            H_out, W_out,
            1, 1, 1, 1,  # stride_H, stride_W, pad_H, pad_W
            1e-05, 0.01,  # eps, negative_slope
            BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
        )
    
    return output

def replacement_func():
    return conv_bn_relu_fusion