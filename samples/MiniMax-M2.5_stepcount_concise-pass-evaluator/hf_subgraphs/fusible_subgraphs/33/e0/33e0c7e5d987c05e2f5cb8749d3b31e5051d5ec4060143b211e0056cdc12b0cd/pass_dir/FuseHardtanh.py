import torch
import triton
import triton.language as tl

# Pattern matching function for Hardtanh
def pattern(in_3):
    """
    Match Hardtanh operation with specific min_val=0.0, max_val=6.0
    """
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return tmp_3

# Argument extraction function
def replacement_args(in_3):
    return (in_3,)

# Autotune configurations for different tensor sizes
@triton.autotune(
    configs=[
        # Very small tensor (batch=1, 96x56x56 = 301k elements)
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=4),
        # Large tensor (batch=128, 96x64x64 = 50M elements)
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def hardtanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for Hardtanh activation: clamp(x, min_val, max_val)"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Hardtanh: clamp(x, min_val, max_val) using tl.clamp
    x_clamped = tl.clamp(x, min_val, max_val)
    
    # Store result
    tl.store(output_ptr + offsets, x_clamped, mask=mask)

@torch.fx.wrap
def optimized_hardtanh(input):
    """
    Optimized Hardtanh using Triton kernel with autotuning.
    Hardtanh: clamp(x, 0.0, 6.0)
    """
    # Get tensor shape
    n_elements = input.numel()
    output = torch.empty_like(input)
    
    # Calculate grid - use minimum needed blocks
    # Use a single program for small tensors to reduce launch overhead
    # The autotuner will select the best block size
    num_programs = (n_elements + 4096 - 1) // 4096
    num_programs = max(num_programs, 1)
    num_programs = min(num_programs, 64)
    
    # Launch kernel with autotuning
    hardtanh_kernel[(num_programs,)](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        min_val=0.0,
        max_val=6.0,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_hardtanh