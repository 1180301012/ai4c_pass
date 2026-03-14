import torch
import triton
import triton.language as tl

# Pattern to match batch_norm in eval mode with eps=0.001
def pattern(input, mean, var, weight, bias):
    result = torch.nn.functional.batch_norm(input, mean, var, weight, bias, False, 0.1, 0.001)
    return result

def replacement_args(input, mean, var, weight, bias):
    return (input, mean, var, weight, bias)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def batch_norm_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Calculate channel indices for each element
    channel_idx = (offsets // HW) % C
    
    # Load batch norm parameters
    mean_val = tl.load(mean_ptr + channel_idx, mask=mask)
    var_val = tl.load(var_ptr + channel_idx, mask=mask)
    weight_val = tl.load(weight_ptr + channel_idx, mask=mask)
    bias_val = tl.load(bias_ptr + channel_idx, mask=mask)
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute batch norm: y = weight * (x - mean) / sqrt(var + eps) + bias
    inv_std = tl.rsqrt(var_val + eps)
    x_norm = (x - mean_val) * inv_std
    y = weight_val * x_norm + bias_val
    
    # Store output
    tl.store(output_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def batch_norm_triton(input, mean, var, weight, bias):
    # Get tensor shape
    B, C, H, W = input.shape
    N = input.numel()
    HW = H * W
    eps = 0.001
    
    # Ensure all tensors are on GPU and contiguous
    input = input.contiguous()
    device = input.device
    mean = mean.to(device).contiguous()
    var = var.to(device).contiguous()
    weight = weight.to(device).contiguous()
    bias = bias.to(device).contiguous()
    
    # Allocate output
    output = torch.empty_like(input)
    
    # Launch kernel
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    batch_norm_kernel[grid](
        input, output,
        mean, var, weight, bias,
        N, C, HW,
        eps,
    )
    
    return output

def replacement_func():
    return batch_norm_triton