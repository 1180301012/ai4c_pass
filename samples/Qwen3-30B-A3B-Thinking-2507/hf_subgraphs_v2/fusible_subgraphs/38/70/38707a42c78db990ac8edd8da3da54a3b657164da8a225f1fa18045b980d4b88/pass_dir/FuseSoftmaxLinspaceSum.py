import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.softmax(in_0, dim=1)
    tmp_1 = torch.linspace(0, 4, steps=5, device=torch.device('cuda:0'))
    tmp_2 = tmp_0 * tmp_1
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_kernel(
    in_ptr,
    out_ptr,
    B,
    N,
):
    batch_id = tl.program_id(0)
    offsets = tl.arange(0, 5)
    mask = offsets < N
    in_data = tl.load(in_ptr + batch_id * N + offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(in_data, axis=0)
    exp_data = tl.exp(in_data - max_val)
    sum_exp = tl.sum(exp_data, axis=0)
    softmax_data = exp_data / sum_exp
    positions = tl.arange(0, 5)
    weighted = softmax_data * positions
    weighted_sum = tl.sum(weighted, axis=0)
    result = 5.0 - weighted_sum
    tl.store(out_ptr + batch_id, result)

@torch.fx.wrap
def optimized_kernel_wrapper(in_0):
    B = in_0.shape[0]
    N = in_0.shape[1]
    assert N == 5, f"Expected N=5 but got {N}"
    out = torch.empty((B,), dtype=in_0.dtype, device=in_0.device)
    optimized_kernel[(B,)](in_0, out, B, N)
    return out

def replacement_func():
    return optimized_kernel_wrapper