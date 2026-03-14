import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias):
    output = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    return output

def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_inference_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = N * C * H * W
    mask = offsets < total_elements
    
    c_indices = (offsets // (H * W)) % C
    
    mean = tl.load(mean_ptr + c_indices, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_indices, mask=mask, other=0.0)
    weight_val = tl.load(weight_ptr + c_indices, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + c_indices, mask=mask, other=0.0)
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std * weight_val + bias_val
    
    tl.store(output_ptr + offsets, x_norm, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(input, running_mean, running_var, weight, bias):
    N, C, H, W = input.shape
    output = torch.empty_like(input)
    
    eps = 0.001
    BLOCK_SIZE = 1024
    total_elements = N * C * H * W
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    batch_norm_inference_kernel[grid](
        input, output,
        running_mean, running_var, weight, bias,
        N, C, H, W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_batch_norm