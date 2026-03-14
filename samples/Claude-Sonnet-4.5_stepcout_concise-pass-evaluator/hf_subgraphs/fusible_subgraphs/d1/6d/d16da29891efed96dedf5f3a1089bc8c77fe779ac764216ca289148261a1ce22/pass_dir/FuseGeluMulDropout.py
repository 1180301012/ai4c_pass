import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_0, in_1):
    """
    Match the pattern: GELU -> multiply -> dropout
    Must match exactly with model.py including positional arguments
    """
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel without autotuning for lower overhead
@triton.jit
def fused_gelu_mul_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for GELU + elementwise multiplication
    Since dropout with training=False is a no-op, we don't need to implement it
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU(x) using the exact formula
    # GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.4142135623730951
    gelu_out = 0.5 * x * (1.0 + tl.math.erf(x / sqrt_2))
    
    # Multiply with second input
    result = gelu_out * y
    
    # Store result (dropout with training=False is identity, so we skip it)
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_gelu_mul_dropout(in_0, in_1):
    """
    Wrapper function that launches the fused kernel
    """
    # Get total number of elements
    n_elements = in_0.numel()
    
    # Allocate output tensor
    out = torch.empty_like(in_0)
    
    # Use a smaller block size for better small-tensor performance
    BLOCK_SIZE = 512
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel with num_warps=4 for optimal performance
    fused_gelu_mul_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_gelu_mul_dropout