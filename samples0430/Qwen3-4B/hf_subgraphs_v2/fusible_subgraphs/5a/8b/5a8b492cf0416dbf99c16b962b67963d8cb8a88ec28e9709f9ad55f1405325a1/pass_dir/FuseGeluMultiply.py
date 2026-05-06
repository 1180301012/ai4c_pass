import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp1 = tmp0 * in_1
    tmp2 = torch.nn.functional.dropout(tmp1, 0.1, False, False)
    return tmp2

def replacement_args(in_0, in_1):
    return in_0, in_1

@triton.jit
def optimized_kernel(in_0_ptr, in_1_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)

    sqrt2 = 1.41421356237
    z = in_0 * sqrt2
    exp_neg_z = tl.exp(-z)
    sigmoid_z = 1.0 / (1.0 + exp_neg_z)
    gelu = in_0 * sigmoid_z

    out = gelu * in_1

    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    n_elems = in_0.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elems + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)

    optimized_kernel[(num_blocks,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return kernel_wrapper