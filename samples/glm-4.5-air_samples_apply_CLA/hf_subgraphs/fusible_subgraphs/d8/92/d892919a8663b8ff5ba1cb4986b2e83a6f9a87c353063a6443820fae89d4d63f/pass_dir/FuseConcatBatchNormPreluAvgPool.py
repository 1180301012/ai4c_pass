import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    x1_ptr, x2_ptr,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    prelu_weight_ptr,
    prelu_out_ptr, pooled_out_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes one spatial location and channel
    pid = tl.program_id(0)
    
    # Calculate spatial and channel indices
    spatial_pid = pid // C
    c = pid % C
    
    if spatial_pid >= H * W or c >= C:
        return
    
    # Convert spatial index to (h, w)
    h = spatial_pid // W
    w = spatial_pid % W
    
    # Load input data (concatenation is implicit in memory layout)
    if c < C // 2:
        # First half (x1)
        x_off = (h * W + w) * (C // 2) + c
        x1 = tl.load(x1_ptr + x_off, other=0.0)
        x2 = 0.0  # x2 doesn't exist in first half
    else:
        # Second half (x2)
        x_off = (h * W + w) * (C // 2) + (c - C // 2)
        x1 = 0.0  # x1 doesn't exist in second half
        x2 = tl.load(x2_ptr + x_off, other=0.0)
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + c, other=0.0)
    running_var = tl.load(running_var_ptr + c, other=1.0)
    weight = tl.load(weight_ptr + c, other=1.0)
    bias = tl.load(bias_ptr + c, other=0.0)
    prelu_weight = tl.load(prelu_weight_ptr + c, other=1.0)
    
    # Batch normalization
    var = tl.sqrt(running_var + eps)
    x1_norm = (x1 - running_mean) / var * weight + bias
    x2_norm = (x2 - running_mean) / var * weight + bias
    
    # PReLU activation
    x1_out = tl.where(x1_norm < 0, x1_norm * prelu_weight, x1_norm)
    x2_out = tl.where(x2_norm < 0, x2_norm * prelu_weight, x2_norm)
    
    # Store PReLU output
    prelu_off = (h * W + w) * C + c
    if c < C // 2:
        tl.store(prelu_out_ptr + prelu_off, x1_out)
    else:
        tl.store(prelu_out_ptr + prelu_off, x2_out)

@triton.jit
def global_avg_pool_kernel(
    input_ptr, output_ptr,
    N, H, W, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= N * C:
        return
    
    # Global average pooling across spatial dimensions (H x W)
    total = 0.0
    spatial_elements = H * W
    
    # Sum over all spatial locations for this (n, c) combination
    for h in range(H):
        for w in range(W):
            offset = (h * W + w) * C + (pid % C)
            # Only add if this n index matches
            if h * W + w < spatial_elements:
                x = tl.load(input_ptr + offset, other=0.0)
                total += x
    
    # Store average
    avg_val = total / spatial_elements
    tl.store(output_ptr + pid, avg_val)

@torch.fx.wrap
def fused_op(x1, x2, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    N, C1, H, W = x1.shape
    C2 = x2.shape[1]
    C_total = C1 + C2
    
    # Create output tensors using existing tensor shapes to avoid proxy issues
    prelu_output = torch.empty_like(x1.expand(N, C_total, H, W))
    pooled_output = torch.empty_like(x1.view(N, C1))  # Use similar shape for pooled output
    
    # Launch fused kernel for batch_norm + prelu
    fused_grid_size = H * W * C_total
    fused_kernel[fused_grid_size](
        x1, x2,
        running_mean, running_var, bn_weight, bn_bias, prelu_weight,
        prelu_output, pooled_output,
        N, C_total, H, W,
        1e-05,
        128,
    )
    
    # Global average pooling using triton kernel
    pool_grid_size = N * C_total
    global_avg_pool_kernel[pool_grid_size](
        prelu_output, pooled_output,
        N, H, W, C_total,
        256,
    )
    
    # Return the structure expected by the pattern: prelu output and pooled view
    return prelu_output, pooled_output.view(N, C_total)

def pattern(x1, x2, bn_running_mean, bn_running_var, bn_weight, bn_bias, prelu_weight):
    # Pattern matches the exact computation structure from original graphs
    tmp_0 = prelu_weight
    tmp_1 = bn_running_mean
    tmp_2 = bn_running_var
    tmp_3 = bn_bias
    tmp_4 = bn_weight
    tmp_5 = torch.cat([x1, x2], 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
    tmp_7 = torch.prelu(tmp_6, tmp_0)
    tmp_8 = torch.nn.functional.adaptive_avg_pool2d(tmp_7, 1)
    tmp_9 = tmp_8.view(tmp_8.size(0), tmp_8.size(1))  # Dynamic view to match both graphs
    return tmp_7, tmp_9

def replacement_args(x1, x2, bn_running_mean, bn_running_var, bn_weight, bn_bias, prelu_weight):
    return (x1, x2, bn_running_mean, bn_running_var, bn_weight, bn_bias, prelu_weight)

def replacement_func():
    return fused_op