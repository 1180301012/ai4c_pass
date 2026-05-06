import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def silu_mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    inv_x = tl.exp(-x)
    sigmoid_x = inv_x / (1 + inv_x)
    silu_out = x * sigmoid_x

    out = silu_out * y

    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    n = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)
    silu_mul_kernel[(num_programs,) ](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return kernel_wrapper