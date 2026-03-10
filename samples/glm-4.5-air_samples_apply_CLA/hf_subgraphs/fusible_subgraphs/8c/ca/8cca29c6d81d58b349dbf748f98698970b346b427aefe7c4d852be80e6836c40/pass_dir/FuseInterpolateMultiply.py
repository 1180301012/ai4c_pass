import torch
import triton
import triton.language as tl


def pattern(a, b):
    """Match multiply pattern.
    
    This matches: a * b
    """
    out = a * b
    return out


def replacement_args(a, b):
    return (a, b)


# Simple elementwise multiply kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['batch_size', 'channels', 'height', 'width']
)
@triton.jit
def mul_kernel(
    a_ptr, b_ptr, output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    """Elementwise multiply kernel."""
    pid = tl.program_id(0)
    num_programs = grid[0]
    
    total_elements = batch_size * channels * height * width
    elements_per_program = (total_elements + num_programs - 1) // num_programs
    start_elem = pid * elements_per_program
    end_elem = min(start_elem + elements_per_program, total_elements)
    
    for idx in range(start_elem, end_elem):
        # Load and compute
        a_val = tl.load(a_ptr + idx)
        b_val = tl.load(b_ptr + idx)
        result = a_val * b_val
        tl.store(output_ptr + idx, result)


@torch.fx.wrap
def fused_multiply(a, b):
    """Fused multiply kernel wrapper."""
    # Get shapes
    B, C, H, W = a.shape
    
    # Allocate output tensor
    out = torch.empty_like(a)
    
    # Calculate grid
    total_elements = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    mul_kernel[(num_programs,)](
        a, b, out,
        B, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


def replacement_func():
    return fused_multiply