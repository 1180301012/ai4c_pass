import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
def pattern(in_0, in_1, in_2):
    """
    Match the SiLU activation with detach operations.
    The pattern computes:
    - SiLU activation on in_0
    - Detach in_1, in_2, and the SiLU output
    Returns all 4 values in the correct order.
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel for SiLU activation
@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for SiLU activation"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    neg_x = -x
    sigmoid = 1.0 / (1.0 + tl.exp(neg_x))
    out = x * sigmoid
    
    # Store
    tl.store(output_ptr + offsets, out, mask=mask)


def silu_kernel_wrapper(in_0, out_0):
    """Wrapper to launch the Triton kernel"""
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    silu_kernel[(num_programs,)](
        in_0,
        out_0,
        n_elements,
        BLOCK_SIZE,
    )


# Use torch.fx.wrap to make the Triton kernel call traceable
@torch.fx.wrap
def silu_kernel_fx(in_0, out_0):
    silu_kernel_wrapper(in_0, out_0)


def optimized_silu(in_0, in_1, in_2):
    """
    Optimized implementation using Triton kernel for SiLU.
    The detach operations are metadata operations and don't need computation.
    """
    # Allocate output for SiLU result
    out_0 = torch.empty_like(in_0)
    
    # Launch Triton kernel for SiLU activation
    silu_kernel_fx(in_0, out_0)
    
    # The detach operations are metadata - they don't copy data
    # They just create views that don't require grad
    out_1 = in_1.detach()
    out_2 = in_2.detach()
    out_3 = out_0.detach()
    
    return (out_1, out_2, out_3, out_0)


def replacement_func():
    return optimized_silu