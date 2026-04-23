import torch
import triton
import triton.language as tl

# Pattern matching function - matching the exact operations in the model
# Note: Must not include cleanup statements like 'tmp_x = None'
def pattern(tmp_1, in_1):
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3

# Extract arguments for replacement
# Returns exactly the arguments needed to compute the result
# Must match the order in pattern function

def replacement_args(tmp_1, in_1):
    return (tmp_1, in_1)

# Triton kernel for elementwise multiplication with broadcast (equivalent to in_1 * tmp_1)
@triton.jit
def elementwise_mul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper with grid setup
@torch.fx.wrap
def elementwise_mul(a, b):
    N = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(b)
    elementwise_mul_kernel[(num_programs,)](a, b, out, N, BLOCK_SIZE)
    return out

# Return the kernel function reference (not call it)
def replacement_func():
    return elementwise_mul