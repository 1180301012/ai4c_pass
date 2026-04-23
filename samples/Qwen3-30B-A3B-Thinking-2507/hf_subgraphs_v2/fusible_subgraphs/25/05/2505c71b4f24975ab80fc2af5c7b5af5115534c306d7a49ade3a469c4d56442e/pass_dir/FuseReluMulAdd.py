import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_2, in_1, in_0):
    return (in_0, in_1, in_2)

@triton.jit
def elementwise_kernel(
    in_2_ptr,
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in_0_val = tl.load(in_0_ptr)
    in_1_val = tl.load(in_1_ptr)

    x = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    x = x * in_1_val
    x = x + in_0_val

    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fuse_relu_mul_add(in_0, in_1, in_2):
    n_elements = in_2.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_2)

    elementwise_kernel[(num_blocks,)](
        in_2_ptr=in_2,
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

def replacement_func():
    return fuse_relu_mul_add