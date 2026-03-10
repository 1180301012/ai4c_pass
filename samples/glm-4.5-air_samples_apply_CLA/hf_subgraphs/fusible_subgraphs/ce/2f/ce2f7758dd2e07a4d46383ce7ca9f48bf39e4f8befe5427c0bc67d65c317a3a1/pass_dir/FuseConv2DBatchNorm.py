import torch
import triton
import triton.language as tl

def pattern(x, weight, running_mean, running_var, bn_weight, bn_bias):
    # Conv2D operation with proper stride, padding, and dilation
    conv_out = torch.conv2d(x, weight, None, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    # BatchNorm operation
    bn_out = torch.nn.functional.batch_norm(conv_out, running_mean, running_var, bn_weight, bn_bias, training=False, momentum=0.1, eps=1e-05)
    return conv_out, bn_out

@triton.jit
def conv2d_kernel(
    x_ptr, weight_ptr, running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    conv_out_ptr, bn_out_ptr,
    N, C_in, H, W, K,
    pad: tl.constexpr, stride: tl.constexpr, kernel_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # batch dimension  
    pid_n = tl.program_id(1)  # output channel dimension
    
    base_idx = pid_m * K * H * W + pid_n * H * W
    
    for h in range(H):
        for w in range(W):
            # Compute convolution result
            conv_val = 0.0
            
            # 3x3 convolution with stride=1, padding=1
            for k_h in range(-pad, kernel_size - pad, stride):
                for k_w in range(-pad, kernel_size - pad, stride):
                    h_in = h + k_h
                    w_in = w + k_w
                    
                    # Handle bounds (padding with zero)
                    if 0 <= h_in < H and 0 <= w_in < W:
                        # Load input value with proper channel offset
                        x_offset = pid_m * C_in * H * W + h_in * W + w_in
                        x_val = tl.load(x_ptr + x_offset, other=0.0)
                        
                        # Load weight for [output_channel, input_channel, h, w]
                        kernel_idx = (k_h + pad) * 3 + (k_w + pad)
                        weight_offset = n * C_in * 9 + kernel_idx
                        weight_val = tl.load(weight_ptr + weight_offset, other=0.0)
                        
                        conv_val += x_val * weight_val
            
            # Load batch norm parameters
            bn_mean = tl.load(running_mean_ptr + n, other=0.0)
            bn_var = tl.load(running_var_ptr + n, other=1.0)
            bn_w = tl.load(bn_weight_ptr + n, other=1.0)
            bn_b = tl.load(bn_bias_ptr + n, other=0.0)
            
            # Compute batch normalization
            scale = bn_w / tl.math.sqrt(bn_var + 1e-05)
            bn_val = (conv_val - bn_mean) * scale + bn_b
            
            # Store results
            out_idx = base_idx + h * W + w
            tl.store(conv_out_ptr + out_idx, conv_val)
            tl.store(bn_out_ptr + out_idx, bn_val)

@torch.fx.wrap
def fused_conv_bn(x, weight, running_mean, running_var, bn_weight, bn_bias):
    # Get tensor shapes
    N, C_in, H_in, W_in = x.shape
    K, C_out, k_h, k_w = weight.shape
    H_out, W_out = H_in, W_in  # With padding=1, stride=1 for 3x3 kernel
    
    # Create output tensors
    conv_out = torch.empty((N, K, H_out, W_out), dtype=torch.float32, device=x.device)
    bn_out = torch.empty((N, K, H_out, W_out), dtype=torch.float32, device=x.device)
    
    # Launch parameters
    BLOCK_SIZE_M = 1  # Process one batch element per program
    BLOCK_SIZE_N = 32  # Process 32 output channels per program
    
    # Grid size
    grid_x = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_y = (K + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_z = 1
    
    # Launch kernel
    conv2d_kernel[(grid_x, grid_y, grid_z)](
        x, weight, running_mean, running_var, bn_weight, bn_bias,
        conv_out, bn_out,
        N, C_in, H_in, W_in, K,
        1, 1, 3,  # pad=1, stride=1, kernel_size=3
        BLOCK_SIZE_M, BLOCK_SIZE_N,
    )
    
    return conv_out, bn_out



def replacement_args(x, weight, running_mean, running_var, bn_weight, bn_bias):
    return (x, weight, running_mean, running_var, bn_weight, bn_bias)

def replacement_func():
    return fused_conv_bn