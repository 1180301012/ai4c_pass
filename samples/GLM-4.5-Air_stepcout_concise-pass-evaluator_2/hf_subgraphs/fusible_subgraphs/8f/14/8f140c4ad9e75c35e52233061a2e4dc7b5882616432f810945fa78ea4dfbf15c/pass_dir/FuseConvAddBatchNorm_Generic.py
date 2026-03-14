import torch
import triton
import triton.language as tl
import math

def pattern(in_6, tmp_5, tmp_4, in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    # Conv2D operation - use the actual groups parameter from the weight tensor shape
    # tmp_5 has shape [C, 1, K, K] where C is the number of groups
    groups_192 = tmp_5.shape[0]  # Extract groups count from weight tensor shape
    tmp_6 = torch.conv2d(in_6, tmp_5, tmp_4, (1, 1), (3, 3), (1, 1), groups_192)
    # Addition operation  
    tmp_7 = in_7 + tmp_6
    # BatchNorm operation
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return tmp_6, tmp_7, tmp_8

def replacement_args(in_6, tmp_5, tmp_4, in_7, tmp_0, tmp_1, tmp_3, tmp_2):
    return (in_6, tmp_5, tmp_4, in_7, tmp_0, tmp_1, tmp_3, tmp_2)

@triton.jit
def fused_conv_add_norm_kernel(
    input_ptr,           # Input tensor [N, C, H, W]
    weight_ptr,          # Conv weight [C, 1, K, K] 
    bias_ptr,            # Conv bias [C]
    residual_ptr,        # Residual tensor [N, C, H, W]
    running_mean_ptr,    # Batch norm running mean [C]
    running_var_ptr,     # Batch norm running var [C]
    weight_norm_ptr,     # Batch norm weight [C]
    bias_norm_ptr,       # Batch norm bias [C]
    output_ptr,          # Output tensor [N, C, H, W]
    N, C, H, W, K,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate indices
    n_start = pid_n * BLOCK_SIZE_N
    c_offset = pid_c * BLOCK_SIZE_C
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    
    # Process one tile
    for n in range(n_start, min(n_start + BLOCK_SIZE_N, N)):
        for c in range(c_offset, min(c_offset + BLOCK_SIZE_C, C)):
            # Load batch norm parameters
            running_mean = tl.load(running_mean_ptr + c)
            running_var = tl.load(running_var_ptr + c)
            gamma = tl.load(weight_norm_ptr + c)
            beta = tl.load(bias_norm_ptr + c)
            conv_bias = tl.load(bias_ptr + c)
            
            # Process spatial region
            for h in range(h_start, min(h_start + BLOCK_SIZE_H, H)):
                for w in range(w_start, min(w_start + BLOCK_SIZE_W, W)):
                    # Convolution with padding=(3,3) for 7x7 kernel
                    conv_val = conv_bias
                    kh_start = max(0, 3 - h)
                    kh_end = min(K, 3 + (H - h))
                    kw_start = max(0, 3 - w)
                    kw_end = min(K, 3 + (W - w))
                    
                    for kh in range(kh_start, kh_end):
                        for kw in range(kw_start, kw_end):
                            h_in = h + kh - 3
                            w_in = w + kw - 3
                            if 0 <= h_in < H and 0 <= w_in < W:
                                input_val = tl.load(input_ptr + n * C * H * W + c * H * W + h_in * W + w_in)
                                # Access weight for channel c at position (kh, kw)
                                weight_offset = c * (1 * K * K)
                                weight_val = tl.load(weight_ptr + weight_offset + kh * K + kw)
                                conv_val += input_val * weight_val
                    
                    # Add residual connection
                    residual_val = tl.load(residual_ptr + n * C * H * W + c * H * W + h * W + w)
                    add_val = conv_val + residual_val
                    
                    # Batch normalization
                    normalized_val = (add_val - running_mean) / tl.sqrt(running_var + eps)
                    bn_val = normalized_val * gamma + beta
                    
                    # Store result
                    tl.store(output_ptr + n * C * H * W + c * H * W + h * W + w, bn_val)

@torch.fx.wrap
def fused_conv_add_norm(input, weight, bias, residual, running_mean, running_var, weight_norm, bias_norm):
    N, C, H, W = input.shape
    K = 7  # Kernel size
    
    # Calculate optimal block sizes
    BLOCK_SIZE_N = 4   # Process multiple batches together
    BLOCK_SIZE_C = 32  # Process multiple channels together
    BLOCK_SIZE_H = 8   # Height tile size
    BLOCK_SIZE_W = 8   # Width tile size
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    output = torch.empty_like(input)
    
    # Launch kernel with 4D grid
    fused_conv_add_norm_kernel[(grid_n, grid_c, grid_h, grid_w)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        residual_ptr=residual,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_norm_ptr=weight_norm,
        bias_norm_ptr=bias_norm,
        output_ptr=output,
        N=N, C=C, H=H, W=W, K=K,
        eps=1e-05,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return output

def replacement_func():
    return fused_conv_add_norm