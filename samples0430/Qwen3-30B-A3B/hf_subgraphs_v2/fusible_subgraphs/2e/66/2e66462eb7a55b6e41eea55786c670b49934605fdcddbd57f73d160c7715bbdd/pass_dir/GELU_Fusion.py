import torch
import triton
import triton.language as tl

def pattern(x):
    tmp0 = 0.5 * x
    tmp1 = torch.pow(x, 3.0)
    tmp2 = 0.044715 * tmp1
    tmp3 = x + tmp2
    tmp4 = 0.7978845608028654 * tmp3
    tmp5 = torch.tanh(tmp4)
    tmp6 = 1.0 + tmp5
    tmp7 = tmp0 * tmp6
    return tmp7

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    x_cubed = x * x * x
    term = x + 0.044715 * x_cubed
    scaled = 0.7978845608028654 * term
    tanh_val = tl.tanh(scaled)
    result = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def gelu_kernel_wrapper(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    optimized_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return gelu_kernel_wrapper