import torch
import triton
import triton.language as tl


def pattern(x):
    y = torch.nn.functional.dropout(x, 0.0, False, False)
    return y


def replacement_args(x):
    return (x,)


@triton.jit
def _identity_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x, mask=mask)


@torch.fx.wrap
def eliminate_dropout_p0_eval(x):
    return x


def replacement_func():
    return eliminate_dropout_p0_eval