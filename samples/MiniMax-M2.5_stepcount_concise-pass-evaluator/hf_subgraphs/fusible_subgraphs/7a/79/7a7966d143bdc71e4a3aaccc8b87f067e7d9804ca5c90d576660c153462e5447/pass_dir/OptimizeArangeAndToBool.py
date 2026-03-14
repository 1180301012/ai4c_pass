import torch
from torch import device
import triton
import triton.language as tl


# Pattern matching function - matches the .to() operation with device
def pattern(in_0):
    """
    Match the .to(device=device(...), dtype=torch.bool) operation
    """
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# Triton kernel for efficient int64 to bool conversion
@triton.jit
def to_bool_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load int64 values and convert to bool (non-zero -> True, zero -> False)
    values = tl.load(input_ptr + offsets, mask=mask, other=0)
    bool_values = values != 0
    tl.store(output_ptr + offsets, bool_values, mask=mask)


@torch.fx.wrap
def to_bool_wrapper(input_tensor):
    # Simply convert dtype - PyTorch will handle it efficiently
    return input_tensor.to(dtype=torch.bool)


# Triton kernel for arange
@triton.jit
def arange_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, offsets, mask=mask)


@torch.fx.wrap
def arange_wrapper(n_elements, device):
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty(n_elements, dtype=torch.int64, device=device)
    arange_kernel[(num_programs,)](
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def replacement_func():
    return to_bool_wrapper