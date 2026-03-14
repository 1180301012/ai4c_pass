import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_0, in_1, in_2, in_3):
    # Match batch_norm pattern
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_6

def replacement_args(tmp_5, in_0, in_1, in_2, in_3):
    return (tmp_5, in_0, in_1, in_2, in_3)

@triton.jit
def batch_norm_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    C, H, W, eps: tl.constexpr
):
    pid = tl.program_id(0)
    
    if pid >= C:
        return
    
    # Pre-compute batch norm parameters for this channel
    beta = tl.load(bias_ptr + pid)
    gamma = tl.load(weight_ptr + pid)
    mean = tl.load(running_mean_ptr + pid)
    var = tl.load(running_var_ptr + pid)
    
    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Process all spatial positions for this channel
    for i in range(H * W):
        input_idx = pid * H * W + i
        input_val = tl.load(input_ptr + input_idx)
        
        # Apply batch normalization
        normalized = gamma * (input_val - mean) * inv_std + beta
        
        output_idx = pid * H * W + i
        tl.store(output_ptr + output_idx, normalized)

@torch.fx.wrap
def optimized_batch_norm(input, running_mean, running_var, weight, bias, 
                        training=False, momentum=0.1, eps=1e-05):
    B, C, H, W = input.shape
    N = B * H * W
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Triton launch configuration
    num_channels = (C + 127) // 128  # Use 128 channel blocks
    
    # Launch kernel
    batch_norm_kernel[(num_channels,)](
        input, running_mean, running_var, weight, bias, output,
        C, H, W, eps
    )
    
    return output

def replacement_func():
    return optimized_batch_norm