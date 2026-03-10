import torch
import triton
import triton.language as tl

def pattern(in_4, in_5, in_0, in_1, in_3, in_2):
    """
    Pattern: Match the core computation including identity dropout operations
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    # Dropout with p=0.0 are identity operations
    tmp_6 = tmp_5  # Identity: dropout(tmp_5, 0.0, False, False) 
    tmp_7 = tmp_6  # Identity: dropout(tmp_6, 0.0, False, False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_8, tmp_7

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Extract arguments for the fused kernel
    """
    # Pattern expects: in_4, in_5, in_0, in_1, in_3, in_2
    return in_4, in_5, in_0, in_1, in_3, in_2

@triton.jit
def fused_add_mean_batchnorm_kernel(
    x1_ptr, x2_ptr,
    mean_reduced_ptr,
    batch_normed_ptr,
    batch_size, channels, height, width,
    running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Element-wise addition of two tensors
    2. Mean reduction over spatial dimensions (H, W)
    3. Batch normalization on reduced tensor
    """
    # Program grid indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Compute mean for this batch and channel
    sum_val = 0.0
    count = height * width
    
    # Process spatial dimensions in blocks
    for h in range(0, height):
        for w in range(0, width):
            offset = ((batch_idx * channels + channel_idx) * height + h) * width + w
            x1_val = tl.load(x1_ptr + offset, mask=True, other=0.0)
            x2_val = tl.load(x2_ptr + offset, mask=True, other=0.0)
            sum_val += x1_val + x2_val
    
    # Compute mean
    mean_val = sum_val / count
    
    # Store mean result
    mean_offset = batch_idx * channels + channel_idx
    tl.store(mean_reduced_ptr + mean_offset, mean_val)
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + channel_idx)
    running_var = tl.load(running_var_ptr + channel_idx)
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    # Apply batch normalization
    batch_normed_val = (mean_val - running_mean) / tl.sqrt(running_var + eps)
    batch_normed_val = batch_normed_val * weight_val + bias_val
    
    # Store batch norm result
    tl.store(batch_normed_ptr + mean_offset, batch_normed_val)

@torch.fx.wrap
def fused_add_mean_batchnorm(in_4, in_5, in_0, in_1, in_3, in_2):
    """
    High-performance fused implementation that optimizes identity dropout operations
    """
    # Extract tensor references from the parameters
    x1, x2 = in_4, in_5
    running_mean, running_var = in_0, in_1
    weight, bias = in_3, in_2
    
    B, C, H, W = x1.shape
    
    # Create output tensors
    batch_normed = torch.empty((B, C), dtype=torch.float32, device=x1.device)
    mean_reduced = torch.empty((B, C), dtype=torch.float32, device=x1.device)  # This is tmp_5, which equals tmp_7 due to identity operations
    
    # Launch fused kernel that skips the identity dropout operations
    grid = (B, C)
    
    fused_add_mean_batchnorm_kernel[grid](
        x1, x2,
        mean_reduced,
        batch_normed,
        B, C, H, W,
        running_mean, running_var, weight, bias,
        eps=1e-05,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
    )
    
    return batch_normed, mean_reduced

def replacement_func():
    """
    Return the fused kernel function
    """
    return fused_add_mean_batchnorm