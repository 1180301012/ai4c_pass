import torch
import triton
import triton.language as tl

# Pattern matching function - matches GELU followed by mean over dims (2,3)
def pattern(x):
    tmp_0 = torch.nn.functional.gelu(x)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_1

# Argument extraction function
def replacement_args(x):
    return (x,)


# Fast GELU kernel using tanh approximation - single configuration for stability
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU tanh approximation - optimized form
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_sq = x * x
    inner = 0.7978845608028654 * x * (1.0 + 0.044715 * x_sq)
    
    # tanh via (exp(2y) - 1) / (exp(2y) + 1)
    exp_2y = tl.exp(2.0 * inner)
    tanh_val = (exp_2y - 1.0) / (exp_2y + 1.0)
    result = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_gelu(x):
    """Compute GELU using Triton kernel"""
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Fixed block size - good balance for various sizes
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    gelu_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@torch.fx.wrap
def compute_mean(x):
    """Compute mean over spatial dimensions"""
    return x.mean((2, 3), keepdim=True)


def gelu_mean_replacement(x):
    gelu_out = triton_gelu(x)
    mean_out = compute_mean(gelu_out)
    return gelu_out, mean_out


def replacement_func():
    return gelu_mean_replacement