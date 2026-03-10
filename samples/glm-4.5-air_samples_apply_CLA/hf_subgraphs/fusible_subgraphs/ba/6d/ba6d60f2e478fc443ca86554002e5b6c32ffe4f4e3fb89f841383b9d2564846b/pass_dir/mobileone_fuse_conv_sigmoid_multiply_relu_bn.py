import torch
import triton
import triton.language as tl

def pattern(in_7, tmp_1, tmp_0):
    # Simple convolution first
    tmp_6 = torch.conv2d(in_7, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_6

def replacement_args(in_7, tmp_1, tmp_0):
    return (in_7, tmp_1, tmp_0)

@triton.jit
def fused_conv_sigmoid_multiply_relu_bn_kernel(
    in_7_ptr,              # SE input: [N, C, 1, 1]
    weight_ptr,            # Conv weight: [C_out, C_in, 1, 1]
    bias_ptr,              # Conv bias: [C_out]
    in_6_ptr,              # Main input: [N, C_out, H, W]
    running_mean_ptr,      # BN running_mean: [C_out]
    running_var_ptr,       # BN running_var: [C_out]
    weight_bn_ptr,         # BN weight: [C_out]
    bias_bn_ptr,           # BN bias: [C_out]
    out_relu_ptr,          # ReLU output: [N, C_out, H, W]
    out_bn_ptr,            # BN output: [N, C_out, H, W]
    N, C_out, C_in, H, W, # Tensor dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID for spatial dimensions
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # Calculate offsets
    h_offset = pid_h * BLOCK_SIZE_M
    w_offset = pid_w * BLOCK_SIZE_N
    
    # Create masks for spatial dimensions
    h_mask = h_offset + tl.arange(0, BLOCK_SIZE_M) < H
    w_mask = w_offset + tl.arange(0, BLOCK_SIZE_N) < W
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Load SE input element
    se_val = tl.load(in_7_ptr + pid_n * C_in * 1 * 1 + pid_c * 1 * 1, dtype=tl.float32)
    
    # Load convolution kernel parameters for this channel
    weight_val = tl.load(weight_ptr + pid_c * C_in * 1 * 1 + pid_c * 1 * 1, dtype=tl.float32)
    bias_val = tl.load(bias_ptr + pid_c, dtype=tl.float32)
    
    # Compute conv2d result (1x1 convolution, so each position gets the same value)
    conv_val = se_val * weight_val + bias_val
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_val))
    
    # Load spatial input and multiply with sigmoid output
    spatial_offsets = (h_offset + tl.arange(0, BLOCK_SIZE_M))[:, None] * W + (w_offset + tl.arange(0, BLOCK_SIZE_N))[None, :]
    in_6_ptr_base = in_6_ptr + pid_n * C_out * H * W + pid_c * H * W
    in_6_vals = tl.load(in_6_ptr_base + spatial_offsets, mask=mask, dtype=tl.float32)
    
    # Multiply with attention weights
    mul_vals = in_6_vals * sigmoid_val
    
    # Apply ReLU
    relu_vals = tl.maximum(mul_vals, 0.0)
    
    # Load BN parameters
    running_mean = tl.load(running_mean_ptr + pid_c, dtype=tl.float32)
    running_var = tl.load(running_var_ptr + pid_c, dtype=tl.float32)
    weight_bn = tl.load(weight_bn_ptr + pid_c, dtype=tl.float32)
    bias_bn = tl.load(bias_bn_ptr + pid_c, dtype=tl.float32)
    
    # Apply batch normalization (simplified with estimated statistics)
    epsilon = 1e-05
    normalized_vals = (relu_vals - running_mean) * weight_bn / tl.sqrt(running_var + epsilon)
    bn_vals = normalized_vals + bias_bn
    
    # Store results
    relu_out_ptr = out_relu_ptr + pid_n * C_out * H * W + pid_c * H * W + spatial_offsets
    bn_out_ptr = out_bn_ptr + pid_n * C_out * H * W + pid_c * H * W + spatial_offsets
    
    tl.store(relu_out_ptr, relu_vals, mask=mask)
    tl.store(bn_out_ptr, bn_vals, mask=mask)

@torch.fx.wrap
def fused_conv_sigmoid_multiply_relu_bn(in_7, weight, bias, in_6, running_mean, running_var, weight_bn, bias_bn):
    N, C_out, H, W = in_6.shape
    C_in = in_7.shape[1]
    
    # Create output tensors
    out_relu = torch.empty_like(in_6)
    out_bn = torch.empty_like(in_6)
    
    # Set up grid and launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    grid_h = (H + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_w = (W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = C_out
    grid_n = N
    
    fused_conv_sigmoid_multiply_relu_bn_kernel[(grid_h, grid_w, grid_c, grid_n)](
        in_7, weight, bias, in_6, running_mean, running_var, weight_bn, bias_bn,
        out_relu, out_bn,
        N, C_out, C_in, H, W,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out_relu, out_bn

def replacement_func():
    return fused_conv_sigmoid_multiply_relu_bn