import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    # Simplified pattern: adaptive_pool2d + batch_norm + relu
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    bn_out = torch.nn.functional.batch_norm(pooled, running_mean, running_var, weight, bias)
    relu_out = torch.nn.functional.relu(bn_out, inplace=True)
    return pooled, relu_out

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def fused_bn_relu_global_pool_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_pooled_ptr,
    out_relu_ptr,
    N, C,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fused kernel that performs global average pooling → batch norm → ReLU
    """
    # Program IDs
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Block sizes
    n = pid * BLOCK_SIZE_N
    c_start = pid_c * BLOCK_SIZE_C
    
    # Create masks for bounds checking
    n_mask = n < N
    c_mask = c_start < C
    
    # Load BN parameters
    if c_mask:
        running_mean = tl.load(
            running_mean_ptr + c_start, 
            mask=c_mask, 
            other=0.0
        ).to(tl.float32)
        
        running_var = tl.load(
            running_var_ptr + c_start, 
            mask=c_mask, 
            other=0.0
        ).to(tl.float32)
        
        weight_val = tl.load(
            weight_ptr + c_start, 
            mask=c_mask, 
            other=1.0
        ).to(tl.float32)
        
        bias_val = tl.load(
            bias_ptr + c_start, 
            mask=c_mask, 
            other=0.0
        ).to(tl.float32)
    
    # Compute global average pooling and fused operations for this batch/channel block
    pooled_sum = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    pooled_count = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    pooled_sq_sum = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    # Accumulate sums across spatial dimensions for current N,C block
    for n_idx in range(min(BLOCK_SIZE_N, N - n)):
        global_n = n + n_idx
        spatial_offset = global_n * C * 64  # H=8, W=8, so H*W=64
        
        for c_idx in range(min(BLOCK_SIZE_C, C - c_start)):
            global_c = c_start + c_idx
            channel_offset = global_c * 64
            
            # Load all spatial positions for this channel
            input_ptr = x_ptr + spatial_offset + channel_offset
            x_vals = tl.load(
                input_ptr + tl.arange(0, 64), 
                mask=tl.arange(0, 64) < 64, 
                other=0.0
            ).to(tl.float32)
            
            # Sum for global average pooling
            channel_sum = tl.sum(x_vals)
            channel_sq_sum = tl.sum(x_vals * x_vals)
            
            if n_mask and n_idx < (N - n) and c_mask and c_idx < (C - c_start):
                pooled_sum[c_idx] += channel_sum
                pooled_sq_sum[c_idx] += channel_sq_sum
                pooled_count[c_idx] += 64  # 8x8=64 spatial positions
    
    # Compute global average and apply fused BN + ReLU
    pooled_out = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    relu_out = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    if c_mask:
        for c_idx in range(min(BLOCK_SIZE_C, C - c_start)):
            if pooled_count[c_idx] > 0:
                # Global average pooling: sum / (H * W)
                channel_mean = pooled_sum[c_idx] / pooled_count[c_idx]
                channel_var = (pooled_sq_sum[c_idx] / pooled_count[c_idx]) - channel_mean * channel_mean
                
                # Batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
                inv_std = tl.math.rsqrt(channel_var + eps)
                bn_output = (channel_mean - running_mean[c_idx]) * inv_std * weight_val[c_idx] + bias_val[c_idx]
                
                # ReLU: max(0, bn_output)
                relu_output = tl.maximum(bn_output, 0.0)
            else:
                channel_mean = 0.0
                bn_output = 0.0
                relu_output = 0.0
            
            pooled_out[c_idx] = channel_mean
            relu_out[c_idx] = relu_output
    
    # Store results
    if pid * BLOCK_SIZE_N < N:
        # Store global average pooling result (shape: [N, C, 1, 1] → [N, C])
        for c_idx in range(min(BLOCK_SIZE_C, C - c_start)):
            if n_mask and c_mask and c_idx < (C - c_start):
                out_idx = n * C + c_start + c_idx
                tl.store(out_pooled_ptr + out_idx, pooled_out[c_idx])
    
    if pid * BLOCK_SIZE_N < N:
        # Store ReLU result (shape: [N, C, 1, 1] → [N, C])
        for c_idx in range(min(BLOCK_SIZE_C, C - c_start)):
            if n_mask and c_mask and c_idx < (C - c_start):
                out_idx = n * C + c_start + c_idx
                tl.store(out_relu_ptr + out_idx, relu_out[c_idx])

@torch.fx.wrap
def fused_bn_relu_global_pool(x, running_mean, running_var, weight, bias, eps=1e-05):
    N, C, H, W = x.shape
    
    # Create output tensors (flattened to [N, C] since pooling reduces to 1x1)
    out_pooled = torch.empty((N, C), dtype=torch.float32, device=x.device)
    out_relu = torch.empty((N, C), dtype=torch.float32, device=x.device)
    
    # Block sizes optimized for global pooling
    BLOCK_SIZE_N = 64  # Batch processing
    BLOCK_SIZE_C = 128  # Channel processing
    
    # Calculate grid
    num_N = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_C = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch fused kernel
    grid = (num_N, num_C)
    fused_bn_relu_global_pool_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out_pooled, out_relu,
        N, C,
        BLOCK_SIZE_N, BLOCK_SIZE_C
    )
    
    # Reshape back to [N, C, 1, 1] for compatibility with original output
    out_pooled = out_pooled.view(N, C, 1, 1)
    out_relu = out_relu.view(N, C, 1, 1)
    
    return out_pooled, out_relu

def replacement_func():
    return fused_bn_relu_global_pool