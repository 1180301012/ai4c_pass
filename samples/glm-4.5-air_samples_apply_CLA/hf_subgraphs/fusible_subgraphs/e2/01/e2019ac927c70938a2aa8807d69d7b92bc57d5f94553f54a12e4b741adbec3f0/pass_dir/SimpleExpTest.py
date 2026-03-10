import torch
import triton
import triton.language as tl

def pattern(in_1):
    # Simple pattern matching just the exponential
    tmp_0 = in_1.exp()
    return tmp_0

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def simple_exp_kernel(
    in_1_ptr,
    out_ptr,
):
    # Simple exponential kernel for scalar
    in_1 = tl.load(in_1_ptr)  # Load scalar directly without mask
    result = tl.exp(in_1)
    tl.store(out_ptr, result)  # Store scalar directly

@torch.fx.wrap
def simple_exp_wrapper(in_1):
    out = torch.empty_like(in_1)
    simple_exp_kernel[(1,)](in_1_ptr=in_1, out_ptr=out)
    return out

def replacement_func():
    return simple_exp_wrapper