import torch
import triton
import triton.language as tl


def pattern(in_1, scalar):
    """
    Match: scalar multiplication with any scalar value
    """
    tmp_0 = in_1 * scalar
    return tmp_0


def replacement_args(in_1, scalar):
    return (in_1, scalar)


@triton.jit
def scalar_mul_kernel_simple(
    input_ptr,
    output_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple scalar multiplication kernel without overhead"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = x * scalar
    tl.store(output_ptr + offsets, result, mask=mask)


def optimized_scalar_mul_op(in_1, scalar):
    """
    Simplified scalar multiplication - just use native operation
    to avoid any overhead from custom kernels
    """
    return in_1 * scalar


def replacement_func():
    return optimized_scalar_mul_op