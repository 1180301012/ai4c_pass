import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.softmax(in_0, dim=1)
    tmp_1 = torch.linspace(0, 4, steps=5, device="cuda:0")
    tmp_2 = tmp_0 * tmp_1
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_kernel(in_0_ptr, out_ptr):
    # Process 5 elements
    total = 0.0
    for idx in range(5):
        x = tl.load(in_0_ptr + idx)
        exp_x = tl.exp(x)
        softmax_x = exp_x / tl.sum(exp_x)
        total += softmax_x * idx
    out = 5 - total
    tl.store(out_ptr, out)

@torch.fx.wrap
def kernel_wrapper(in_0):
    out = torch.zeros(1, dtype=in_0.dtype)
    optimized_kernel[(1,)](in_0_ptr=in_0, out_ptr=out)
    return out
def replacement_func():
    return kernel_wrapper