import torch
import triton
import triton.language as tl

# Pattern matching function - matches just scalar multiplication
def pattern(in_1):
    return in_1 * 0.1767766952966369

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Triton kernel for scalar multiplication
@triton.jit
def scalar_mul_kernel(in_ptr, out_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out_val = x * scale
    tl.store(out_ptr + offsets, out_val, mask=mask)

def optimized_forward(in_1):
    scale = 0.1767766952966369
    # Use empty_like which is allowed by PosionDispatchTensor
    out_0 = torch.empty_like(in_1)
    N = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    scalar_mul_kernel[(num_programs,)](in_1, out_0, scale, N, BLOCK_SIZE)
    return out_0

# Replacement function - returns the replacement function
def replacement_func():
    return optimized_forward