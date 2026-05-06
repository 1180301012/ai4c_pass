import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    view1 = in_1.view(-1, 1)
    mult = view1 * in_2
    view2 = in_0.view(-1, 1)
    expanded = view2.expand_as(mult)
    zeros = mult.new_zeros((1000, 16))
    return (expanded, zeros, mult)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_kernel(
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    start = i * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, n)
    
    for j in range(start, end):
        in_1_val = tl.load(in_1_ptr + j)
        for k in range(d):
            in_2_val = tl.load(in_2_ptr + j * d + k)
            out_val = in_1_val * in_2_val
            tl.store(out_ptr + j * d + k, out_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    n = in_0.numel()
    d = in_2.size(1)
    out = torch.empty((n, d), device=in_0.device, dtype=in_1.dtype)
    zeros = torch.empty((1000, 16), device=in_0.device, dtype=in_1.dtype)
    
    optimized_kernel[(n + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        n=n,
        d=d,
        BLOCK_SIZE=128,
    )
    
    expanded = in_0.view(-1, 1).expand_as(out)
    return (expanded, zeros, out)

def replacement_func():
    return kernel_wrapper