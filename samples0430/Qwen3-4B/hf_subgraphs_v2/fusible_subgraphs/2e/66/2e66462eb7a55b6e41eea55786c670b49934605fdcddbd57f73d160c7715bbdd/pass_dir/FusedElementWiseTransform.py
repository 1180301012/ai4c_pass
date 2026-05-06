import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return (tmp_7,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Triton kernel
@triton.jit
def fused_elementwise_kernel(
    in_ptr: tl.float16,
    out_ptr: tl.float16,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, n_elements)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (end - start)
    
    x = tl.load(in_ptr + start + offsets, mask=mask, other=0.0)
    
    a = 0.5 * x
    b = x ** 3.0
    c = 0.044715 * b
    d = x + c
    e = 0.7978845608028654 * d
    f = tl.tanh(e)
    g = 1.0 + f
    h = a * g
    
    tl.store(out_ptr + start + offsets, h, mask=mask)

# Kernel wrapper with @torch.fx.wrap
@torch.fx.wrap
def fused_elementwise_kernel_wrapper(x):
    N = x.numel()
    BLOCK_SIZE = 256
    out = torch.empty_like(x)
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_elementwise_kernel[(num_programs,)](
        x,
        out,
        N,
        BLOCK_SIZE,
    )
    return out

# Replacement function
def replacement_func():
    return fused_elementwise_kernel_wrapper