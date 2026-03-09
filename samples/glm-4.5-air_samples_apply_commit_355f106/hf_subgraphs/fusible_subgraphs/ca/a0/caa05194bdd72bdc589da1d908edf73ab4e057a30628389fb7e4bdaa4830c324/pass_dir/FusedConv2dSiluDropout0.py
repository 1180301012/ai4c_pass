import torch
import triton
import triton.language as tl


# Define pattern as a module class to help with tracing
class SiluPattern(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x, inplace=False)


# Pattern matching function - matches just silu
def pattern(x):
    return torch.nn.functional.silu(x, inplace=False)


# Extract arguments needed for replacement
def replacement_args(x):
    return (x,)


# Optimized Triton kernel for silu activation
@triton.jit
def silu_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU activation kernel: x * sigmoid(x)"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_val = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_val
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def silu_kernel_wrapper(x):
    """
    Wrapper for SiLU activation kernel.
    """
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    silu_kernel[(num_programs,)](
        x, output,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return silu_kernel_wrapper