import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0,)

@triton.jit
def transpose_kernel(
    x_ptr,
    y_ptr,
    n0, n1, n2, n3,
    BLOCK_SIZE: tl.constexpr,
):
    global_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = global_idx < n0 * n1 * n2 * n3

    n2n3 = n2 * n3
    n1n2n3 = n1 * n2n3
    
    i = global_idx // n1n2n3
    rem = global_idx % n1n2n3
    j = rem // n2n3
    rem = rem % n2n3
    k = rem // n3
    l = rem % n3

    new_index = i * n1n2n3 + j * n2n3 + l * n2 + k
    x = tl.load(x_ptr + global_idx, mask=mask, other=0.0)
    tl.store(y_ptr + new_index, x, mask=mask)

@torch.fx.wrap
def transpose_wrapper(x):
    n0, n1, n2, n3 = x.shape
    n_elements = n0 * n1 * n2 * n3
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    transpose_kernel[(num_blocks,)](
        x,
        out,
        n0, n1, n2, n3,
        BLOCK_SIZE
    )
    return out

def replacement_func():
    return transpose_wrapper