import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp1 = tmp0 * in_1
    return tmp1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_silu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))

    out = x * y * sigmoid_x
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_mul(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)
    fused_silu_mul_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return fused_silu_mul