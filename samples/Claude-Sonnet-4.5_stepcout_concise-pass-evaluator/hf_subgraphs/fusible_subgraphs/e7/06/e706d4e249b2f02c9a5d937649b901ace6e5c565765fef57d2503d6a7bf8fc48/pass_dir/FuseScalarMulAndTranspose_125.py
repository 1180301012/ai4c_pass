import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Match: scalar multiplication with 0.125
    """
    tmp_0 = in_1 * 0.125
    return tmp_0


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def scalar_mul_kernel_125(
    input_ptr,
    output_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized scalar multiplication kernel for 0.125"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply by scalar
    result = x * scalar
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def optimized_scalar_mul_125(in_1):
    """
    Optimized scalar multiplication with 0.125
    """
    scalar_value = 0.125
    out = torch.empty_like(in_1)
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    scalar_mul_kernel_125[grid](
        in_1,
        out,
        scalar_value,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return optimized_scalar_mul_125