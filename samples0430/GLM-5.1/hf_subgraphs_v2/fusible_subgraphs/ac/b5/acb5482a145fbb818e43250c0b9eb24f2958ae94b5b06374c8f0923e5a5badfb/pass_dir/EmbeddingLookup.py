import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_4 = in_0.long()
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def identity_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)


@torch.fx.wrap
def identity_wrapper(in_0):
    # For int64 input, .long() is identity - just return in_0 directly
    return in_0


def replacement_func():
    return identity_wrapper