import torch
import triton
import triton.language as tl

# Pattern for bge-base-en-v1.5: slice(:128), expand(1, 128), slice(0:128)
def pattern(in_0, in_1):
    tmp_2 = in_1[slice(None, None, None), slice(None, 128, None)]
    tmp_3 = tmp_2.expand(1, 128)
    tmp_4 = in_0[slice(None, None, None), slice(0, 128, None)]
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for slice copy (available for future optimization)
@triton.jit
def slice_copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, vals, mask=mask)

# Optimized replacement - identical operations to pattern
def fused_slice_expand_slice_128(in_0, in_1):
    # Direct view operations - minimal overhead
    return (in_1[:, :128].expand(1, 128), in_0[:, :128])

def replacement_func():
    return fused_slice_expand_slice_128