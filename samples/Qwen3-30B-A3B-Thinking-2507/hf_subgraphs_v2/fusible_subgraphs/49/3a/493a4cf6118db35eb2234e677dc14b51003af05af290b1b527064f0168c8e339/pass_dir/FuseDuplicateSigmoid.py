import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return tmp_8

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_elements, M, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = row * 196 + tl.arange(0, 256)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    max_val = tl.max(x)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x)
    out = exp_x / sum_exp
    tl.store(out_ptr + offsets, out, mask=offsets < n_elements)
@triton.jit
def subtraction_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 1.0 - x
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def subtraction_wrapper(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    subtraction_kernel[(num_programs,)](x, out, n_elements, BLOCK_SIZE)
    return out
@triton.jit
def multiplication_kernel(x_ptr, y_ptr, out_ptr, n_elements, x_numel, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_idx = offsets % x_numel
    x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def multiplication_wrapper(x, y):
    n_elements = y.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(y)
    multiplication_kernel[(num_programs,)](x, y, out, n_elements, x.numel(), BLOCK_SIZE)
    return out
@triton.jit
def addition_kernel(x_ptr, y_ptr, out_ptr, n_elements, x_numel, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_idx = offsets % x_numel
    x = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def addition_wrapper(x, y):
    n_elements = y.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(y)
    addition_kernel[(num_programs,)](x, y, out, n_elements, x.numel(), BLOCK_SIZE)
    return out

@torch.fx.wrap
def softmax_wrapper(x):
    N = x.numel() // x.shape[-1]
    M = x.shape[-1]
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = N
    out = torch.empty_like(x)
    softmax_kernel[(num_programs,)](x, out, n_elements, M, BLOCK_SIZE)
    return out

@torch.fx.wrap
def sigmoid_wrapper(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    sigmoid_kernel[(num_programs,)](x, out, n_elements, BLOCK_SIZE)
    return out

def optimized_compute(in_0, in_1, in_2):
    tmp_1 = softmax_wrapper(in_2)
    tmp_2 = in_0
    sigmoid_val = sigmoid_wrapper(tmp_2)
    one_minus_sigmoid = subtraction_wrapper(sigmoid_val)
    tmp_5 = multiplication_wrapper(one_minus_sigmoid, in_1)
    tmp_7 = multiplication_wrapper(sigmoid_val, tmp_1)
    tmp_8 = addition_wrapper(tmp_5, tmp_7)
    return tmp_8

def replacement_func():
    return optimized_compute