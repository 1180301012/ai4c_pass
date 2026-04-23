import torch
import triton
import triton.language as tl

@triton.jit
def reduce_mean_var_kernel(
    x_ptr, 
    mean_ptr, 
    var_ptr, 
    C, 
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    cid = tl.program_id(0)
    sum_x = tl.zeros([], dtype=tl.float16)
    sum_x2 = tl.zeros([], dtype=tl.float16)
    for i in range(0, N, 128):
        start = i * C + cid
        x = tl.load(x_ptr + start + tl.arange(0, 128) * C, 
                    mask=(i + tl.arange(0, 128)) < N, 
                    other=0.0)
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
    mean_val = sum_x / N
    var_val = (sum_x2 / N) - (mean_val * mean_val) + eps
    tl.store(mean_ptr + cid, mean_val)
    tl.store(var_ptr + cid, var_val)

@triton.jit
def normalize_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    mean_ptr, 
    var_ptr,
    out_ptr,
    N,
    C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    cid = tl.program_id(0)
    i = tl.program_id(1)
    start = i * 128
    mean_val = tl.load(mean_ptr + cid)
    var_val = tl.load(var_ptr + cid)
    inv_sqrt = 1.0 / tl.sqrt(tl.cast(var_val, tl.float32) + tl.cast(eps, tl.float32))
    weight_val = tl.load(weight_ptr + cid)
    bias_val = tl.load(bias_ptr + cid)
    mask = (start + tl.arange(0, 128)) < N
    offset = start * C + cid
    x = tl.load(x_ptr + offset + tl.arange(0, 128) * C, mask=mask, other=0.0)
    out = (x - mean_val) * inv_sqrt * weight_val + bias_val
    tl.store(out_ptr + offset + tl.arange(0, 128) * C, out, mask=mask)

@torch.fx.wrap
def layer_norm(x, weight, bias, eps=1e-06):
    # Keep eps as float32 for kernel computations
    eps = 1e-6
    B, N, C = x.shape
    mean = torch.empty(C, dtype=x.dtype, device=x.device)
    var = torch.empty(C, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 128
    reduce_mean_var_kernel[(C,)](
        x, mean, var,
        B, N, C,
        eps
    )
    out = torch.empty_like(x)
    num_blocks_BN = (B * N + BLOCK_SIZE - 1) // BLOCK_SIZE
    normalize_kernel[(C, num_blocks_BN)](
        x, weight, bias, mean, var, out,
        B, N, C,
        eps
    )
    return out

def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, weight, bias, eps)

def replacement_func():
    return layer_norm