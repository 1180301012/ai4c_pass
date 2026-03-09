import torch
import triton
import triton.language as tl

@torch.fx.wrap
def fused_mean_batch_norm(x, running_mean, running_var, weight, bias, momentum=0.1, eps=1e-05):
    # Get input dimensions
    N, C, H, W = x.shape
    
    # Cast to proper dtype if needed
    x = x.to(torch.float32)
    running_mean = running_mean.to(torch.float32)
    running_var = running_var.to(torch.float32)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32)
    
    # Create output tensors
    output = torch.empty((C,), dtype=torch.float32, device=x.device)
    mean_output = torch.empty((C,), dtype=torch.float32, device=x.device)
    
    # Launch kernel
    grid = (C,)
    fused_mean_batch_norm_kernel[grid](
        x, 
        running_mean, 
        running_var, 
        weight, 
        bias,
        output,
        mean_output,
        N, C, H, W,
        momentum,
        eps,
        BLOCK_SIZE_C=32,
        BLOCK_SIZE_HW=256,
    )
    
    # For the batch norm operation, we need to return the channel-wise output
    # Mean computation is already done and stored in mean_output
    return output, mean_output

def pattern(x, running_mean, running_var, weight, bias):
    # Mean reduction that feeds into batch norm
    mean_val = x.mean((2, 3), keepdim=False)
    # Batch norm uses the mean (even though dropout is eliminated)
    out = torch.nn.functional.batch_norm(mean_val, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return out, mean_val

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def fused_mean_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_output_ptr,
    N, C, H, W,
    momentum,
    eps,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Each program handles a specific channel
    c = tl.program_id(0)
    
    # Compute total elements for mean normalization
    total_hw = H * W
    
    # Load channel-specific parameters
    running_mean = tl.load(running_mean_ptr + c)
    running_var = tl.load(running_var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    
    # Accumulators for mean computation
    sum_val = 0.0
    
    # Compute spatial mean for this channel
    for hw_offset in range(0, total_hw, BLOCK_SIZE_HW):
        hw_end = min(hw_offset + BLOCK_SIZE_HW, total_hw)
        offsets = hw_offset + tl.arange(0, hw_end - hw_offset)
        mask = offsets < total_hw
        
        # Load input data for this channel across all spatial positions
        input_ptr_c_offset = input_ptr + c * (H * W) + offsets
        x = tl.load(input_ptr_c_offset, mask=mask, other=0.0)
        
        # Accumulate sum for mean computation
        sum_val += tl.sum(x, mask=mask)
    
    # Compute mean
    mean_val = sum_val / total_hw
    
    # Apply batch normalization
    # Normalize: (x - running_mean) / sqrt(running_var + eps)
    normalized = (mean_val - running_mean) / tl.sqrt(running_var + eps)
    # Scale and shift: gamma * normalized + beta
    bn_output = weight * normalized + bias
    
    # Store results
    tl.store(mean_output_ptr + c, mean_val)
    tl.store(output_ptr + c, bn_output)

def replacement_func():
    return fused_mean_batch_norm