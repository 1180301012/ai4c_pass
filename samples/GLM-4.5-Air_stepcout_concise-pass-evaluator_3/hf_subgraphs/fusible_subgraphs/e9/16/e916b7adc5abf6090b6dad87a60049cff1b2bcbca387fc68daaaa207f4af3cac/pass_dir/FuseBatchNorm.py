import torch
import triton
import triton.language as tl

def pattern(in_10, in_3, in_4, in_6, in_5):
    # Match the BatchNorm pattern
    out = torch.nn.functional.batch_norm(in_10, in_3, in_4, in_6, in_5, False, 0.1, 1e-05)
    return out, in_10  # Return both outputs matching the original pattern

def replacement_args(in_10, in_3, in_4, in_6, in_5):
    return (in_10, in_3, in_4, in_6, in_5)

@triton.jit
def fused_batch_norm_kernel(
    input_ptr,        # [N, C, H, W] - input tensor
    running_mean_ptr, # [C] - running mean
    running_var_ptr,  # [C] - running variance
    weight_ptr,       # [C] - weight (gamma)
    bias_ptr,         # [C] - bias (beta)
    output_ptr,       # [N, C, H, W] - output tensor
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate element indices
    pid = tl.program_id(0)
    total_elements = N * C * H * W
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert linear offsets to tensor indices
    n = offsets // (C * H * W)
    remainder = offsets % (C * H * W)
    c = remainder // (H * W)
    h = (remainder // W) % H
    w = remainder % W
    
    # Load input value
    input_idx = (n, c, h, w)
    x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Load normalization parameters
    param_idx = (c,)
    running_mean = tl.load(running_mean_ptr + param_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + param_idx, mask=mask, other=1e-05)
    weight = tl.load(weight_ptr + param_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + param_idx, mask=mask, other=0.0)
    
    # Apply batch normalization
    # y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    eps = 1e-05
    normalized = (x - running_mean) / tl.sqrt(running_var + eps)
    y = normalized * weight + bias
    
    # Store result
    tl.store(output_ptr + input_idx, y, mask=mask)

@torch.fx.wrap
def fused_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    N, C, H, W = input_tensor.shape
    
    # Determine optimal block size
    total_elements = N * C * H * W
    BLOCK_SIZE = 1024  # Can be tuned
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty((N, C, H, W), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set ptrs to None for unused parameters
    running_mean_ptr = running_mean if running_mean is not None else None
    running_var_ptr = running_var if running_var is not None else None
    weight_ptr = weight if weight is not None else None
    bias_ptr = bias if bias is not None else None
    
    # Launch kernel
    fused_batch_norm_kernel[(num_programs,)](
        input_tensor,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        output,
        N, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output, input_tensor  # Return both values matching the original pattern

def replacement_func():
    return fused_batch_norm