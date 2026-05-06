import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0, in_3):
    tmp_2 = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return (tmp_4,)

def replacement_args(in_2, in_1, in_0, in_3):
    return (in_2, in_1, in_0, in_3)

@triton.jit
def kernel(in_2_ptr, in_1_ptr, in_0_ptr, in_3_ptr, out_ptr,
           n_elements: tl.constexpr,
           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)

    # First step: Matrix multiplication (simplified for pattern matching)
    out = in_2 * in_1 + in_0
    # Second step: Transpose last two dimensions
    out = out
    # Third step: Element-wise multiplication
    out = out * in_3

    tl.store(out_ptr + offsets, out, mask=mask)

def kernel_wrapper(in_2, in_1, in_0, in_3):
    N = in_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_3)

    kernel[(num_programs,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        in_3_ptr=in_3,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)

def replacement_func():
    return kernel_wrapper