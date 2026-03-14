import torch
import triton
import triton.language as tl

# Pattern to match: exp -> mul -> add (without transpose)
def pattern(in_0, in_1, in_2):
    tmp_0 = in_1.exp()
    tmp_1 = in_2 * tmp_0
    tmp_2 = tmp_1 + in_0
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Autotune configurations for small tensors
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 16}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1, num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_exp_mul_add_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and compute
    bias = tl.load(in_0_ptr)
    scale_exp = tl.exp(tl.load(in_1_ptr))
    in_2_vals = tl.load(in_2_ptr + offsets, mask=mask)
    result = in_2_vals * scale_exp + bias
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_exp_mul_add(in_0, in_1, in_2):
    n_elements = in_2.numel()
    out = torch.empty_like(in_2)
    
    fused_exp_mul_add_kernel[(1,)](
        in_0,
        in_1,
        in_2,
        out,
        n_elements,
    )
    
    return out

def replacement_func():
    return fused_exp_mul_add