import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """ 
    Match: add
    """
    tmp_0 = in_1 + in_0
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_kernel(
    in_0_ptr, 
    in_1_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: add
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute add
    add_result = in_0 + in_1
    
    # Store output
    tl.store(out_ptr + offsets, add_result, mask=mask)


@torch.fx.wrap
def fused_add(in_0, in_1):
    """
    Fused implementation of add
    """
    n_elements = in_0.numel()
    
    # Output has same shape as input
    out = torch.empty_like(in_0)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_add_kernel[grid](
        in_0, in_1, out,
        n_elements,
    )
    
    return out


def replacement_func():
    return fused_add