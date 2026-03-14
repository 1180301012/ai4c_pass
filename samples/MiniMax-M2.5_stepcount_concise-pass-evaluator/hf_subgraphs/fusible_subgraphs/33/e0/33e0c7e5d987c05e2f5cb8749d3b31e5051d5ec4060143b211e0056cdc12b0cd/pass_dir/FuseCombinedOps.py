import torch
import triton
import triton.language as tl

# Pattern matching function for the entire computation
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the entire computation: Conv2d + Hardtanh
    These are independent operations, but we can optimize them together
    to reduce kernel launch overhead.
    """
    tmp_0 = in_0  # bias
    tmp_1 = in_1  # weight
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return tmp_3, tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Autotune configurations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_hardtanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for Hardtanh: clamp(x, min_val, max_val)"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_clamped = tl.clamp(x, min_val, max_val)
    tl.store(output_ptr + offsets, x_clamped, mask=mask)

@torch.fx.wrap
def optimized_combined(bias, weight, input_conv, input_hardtanh):
    """
    Combined operation: Conv2d + Hardtanh
    For Conv2d, we use PyTorch's optimized implementation (cuDNN)
    For Hardtanh, we use our Triton kernel
    """
    # Conv2d - use PyTorch's optimized implementation
    conv_output = torch.nn.functional.conv2d(input_conv, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Hardtanh - use Triton kernel
    n_elements = input_hardtanh.numel()
    output = torch.empty_like(input_hardtanh)
    
    num_programs = (n_elements + 4096 - 1) // 4096
    num_programs = max(num_programs, 1)
    num_programs = min(num_programs, 64)
    
    fused_hardtanh_kernel[(num_programs,)](
        input_ptr=input_hardtanh,
        output_ptr=output,
        n_elements=n_elements,
        min_val=0.0,
        max_val=6.0,
    )
    
    return output, conv_output

# Replacement function
def replacement_func():
    return optimized_combined