import torch
import triton
import triton.language as tl

@triton.jit
def fused_silu_add_kernel(
    x_ptr,
    silu_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: out = x + silu(silu_tensor)
    where silu(x) = x * sigmoid(x)
    
    This fuses two operations into one kernel:
    1. Compute silu(silu_tensor) = silu_tensor * sigmoid(silu_tensor)
    2. Add the result to x
    """
    # Calculate program ID and block offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x and silu_tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    silu_tensor = tl.load(silu_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid(silu_tensor)
    sigmoid_silu = tl.sigmoid(silu_tensor)
    
    # Compute silu = silu_tensor * sigmoid(silu_tensor)
    silu = silu_tensor * sigmoid_silu
    
    # Compute output = x + silu
    out = x + silu
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def autotune_config():
    """Define autotune configurations for optimal performance."""
    return [
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_stages=1, num_warps=8),
    ]


@triton.autotune(configs=autotune_config(), key=['n_elements'])
@triton.jit
def fused_silu_add_kernel_autotuned(
    x_ptr,
    silu_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Autotuned fused kernel for: out = x + silu(silu_tensor)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    silu_tensor = tl.load(silu_ptr + offsets, mask=mask, other=0.0)
    
    sigmoid_silu = tl.sigmoid(silu_tensor)
    silu = silu_tensor * sigmoid_silu
    out = x + silu
    
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_add_dispatch(x, silu_tensor):
    """
    Dispatch function that launches the fused silu+add kernel.
    """
    N = x.numel()
    
    # Use autotuned kernel for optimal performance
    out = torch.empty_like(x)
    
    # Calculate grid size - use enough blocks to cover all elements
    BLOCK_SIZE = 1024  # Initial guess, autotune will adjust
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch autotuned kernel
    fused_silu_add_kernel_autotuned[(num_programs,)](
        x_ptr=x,
        silu_ptr=silu_tensor,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1):
    """
    Match the pattern: silu(in_1) + in_0
    
    The pattern matches:
    - tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    - tmp_1 = tmp_0 + in_0
    - return (tmp_1,)
    """
    # Note: inplace silu modifies in_1, but we need the value for fusion
    # We create a copy to preserve the pattern matching while allowing fusion
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return tmp_1


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the fused replacement.
    """
    return (in_0, in_1)


def replacement_func():
    """
    Return the fused silu+add function.
    """
    return fused_silu_add_dispatch