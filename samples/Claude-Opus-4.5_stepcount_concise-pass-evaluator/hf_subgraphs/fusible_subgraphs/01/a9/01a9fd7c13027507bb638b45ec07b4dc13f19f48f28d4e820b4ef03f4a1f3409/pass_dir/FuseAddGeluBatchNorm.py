import torch
import triton
import triton.language as tl
import math

# Pattern matching function - batch_norm + identity add (0 + x)
def pattern(bn_input, running_mean, running_var, weight, bias):
    # Batch normalization
    tmp_6 = torch.nn.functional.batch_norm(bn_input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    # Identity (0 + x)
    tmp_7 = 0 + tmp_6
    return tmp_7


def replacement_args(bn_input, running_mean, running_var, weight, bias):
    return (bn_input, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_spatial'],
)
@triton.jit
def fused_bn_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    n_spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C
    
    # Load batch norm parameters for this channel
    mean = tl.load(running_mean_ptr + channel_idx)
    var = tl.load(running_var_ptr + channel_idx)
    gamma = tl.load(weight_ptr + channel_idx)
    beta = tl.load(bias_ptr + channel_idx)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    base_offset = batch_idx * C * n_spatial + channel_idx * n_spatial
    
    for start in range(0, n_spatial, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_spatial
        
        x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        
        # Batch norm: gamma * (x - mean) * inv_std + beta
        result = gamma * (x - mean) * inv_std + beta
        
        tl.store(output_ptr + base_offset + offsets, result, mask=mask)


@torch.fx.wrap
def fused_bn(bn_input, running_mean, running_var, weight, bias):
    shape = bn_input.shape
    N = shape[0]
    C = shape[1]
    n_spatial = 1
    for i in range(2, len(shape)):
        n_spatial *= shape[i]
    
    output = torch.empty_like(bn_input)
    
    num_programs = N * C
    
    fused_bn_kernel[(num_programs,)](
        bn_input,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        N,
        C,
        n_spatial,
        1e-5,
    )
    
    return output


def replacement_func():
    return fused_bn