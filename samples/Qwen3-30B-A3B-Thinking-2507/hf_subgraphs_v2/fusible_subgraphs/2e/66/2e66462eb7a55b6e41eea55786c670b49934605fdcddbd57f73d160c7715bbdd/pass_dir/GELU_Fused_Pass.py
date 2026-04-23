import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_cubed = x * x * x
    term = 0.044715 * x_cubed
    sum1 = x + term
    scaled_sum = 0.7978845608028654 * sum1
    scaled_sum_fp32 = tl.cast(scaled_sum, tl.float32)
    exp_val = tl.exp(-2.0 * scaled_sum_fp32)
    tanh_val = (1.0 - exp_val) / (1.0 + exp_val)
    tanh_val = tl.cast(tanh_val, scaled_sum.dtype)
    out = 0.5 * x * (1 + tanh_val)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def gelu_kernel_wrapper(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    gelu_kernel[(num_programs,)](x, out, n_elements, BLOCK_SIZE)
    return out

def replacement_func():
    return gelu_kernel_wrapper