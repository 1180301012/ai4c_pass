import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp1 = in_1.detach()
    tmp2 = in_2.detach()
    tmp3 = tmp0.detach()
    return (tmp1, tmp2, tmp3, tmp0)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    exp_neg_x = tl.exp(-x)
    sigmoid = 1.0 / (1.0 + exp_neg_x)
    out = x * sigmoid
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    N = in_0.numel()
    BLOCK_SIZE = 256
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(in_0)
    silu_kernel[(num_blocks,)](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return (
        in_1.detach(),
        in_2.detach(),
        out.detach(),
        out
    )

def replacement_func():
    return kernel_wrapper