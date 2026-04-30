import torch
import triton
import triton.language as tl


@triton.jit
def bool_cast_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    # Cast int64 to bool: non-zero → 1, zero → 0 (stored as int8)
    result = (x != 0).to(tl.int8)
    tl.store(out_ptr + offsets, result, mask=mask)