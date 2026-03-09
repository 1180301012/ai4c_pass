import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    tmp_8 = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
    return tmp_9

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_relu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a channel
    pid = tl.program_id(0)
    channel_idx = pid
    
    # Load normalization parameters for this channel
    mean_val = tl.load(mean_ptr + channel_idx, mask=True)
    var_val = tl.load(var_ptr + channel_idx, mask=True)
    weight_val = tl.load(weight_ptr + channel_idx, mask=True)
    bias_val = tl.load(bias_ptr + channel_idx, mask=True)
    
    # Calculate variance with epsilon
    rstd = 1.0 / tl.sqrt(var_val + eps)
    
    # Process spatial dimensions using tiling
    for h in range(0, height, BLOCK_SIZE_M):
        for w in range(0, width, BLOCK_SIZE_N):
            # Calculate tile bounds
            h_end = min(h + BLOCK_SIZE_M, height)
            w_end = min(w + BLOCK_SIZE_N, width)
            
            # Process each tile element
            for out_h in range(h, h_end):
                for out_w in range(w, w_end):
                    # Calculate input location
                    x_idx = channel_idx * height * width + out_h * width + out_w
                    x_val = tl.load(x_ptr + x_idx, mask=True)
                    
                    # Batch normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
                    norm_val = (x_val - mean_val) * rstd
                    bn_val = norm_val * weight_val + bias_val
                    
                    # ReLU
                    relu_val = tl.max(bn_val, 0.0)
                    
                    # Store result
                    out_idx = channel_idx * height * width + out_h * width + out_w
                    tl.store(out_ptr + out_idx, relu_val)

@torch.fx.wrap
def batch_norm_relu_fusion(x, running_mean, running_var, weight, bias):
    B, C, H, W = x.shape
    eps = 0.001
    
    # Calculate grid size
    BLOCK_SIZE_M = 16  # Block size for height
    BLOCK_SIZE_N = 16  # Block size for width
    grid_size = C * B  # One program per channel
    
    out = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
    
    batch_norm_relu_kernel[grid_size](
        x_ptr=x,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        eps=eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    def wrapper(x, running_mean, running_var, weight, bias):
        return batch_norm_relu_fusion(x, running_mean, running_var, weight, bias)
    return wrapper