import torch
import triton
import triton.language as tl

# ============================================================================
# TRITON KERNEL FOR ARANGE
# ============================================================================

@triton.jit
def arange_kernel(
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to generate arange(0, N)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    values = offsets
    tl.store(out_ptr + offsets, values, mask=mask)


# ============================================================================
# WRAPPER FUNCTION
# ============================================================================

@torch.fx.wrap
def triton_arange(N, device_obj):
    """Generate arange(0, N) using Triton kernel"""
    BLOCK_SIZE = 1024
    N_int = int(N)
    num_programs = (N_int + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_programs < 1:
        num_programs = 1
    
    out = torch.empty(N_int, dtype=torch.int64, device=device_obj)
    arange_kernel[(num_programs,)](
        out_ptr=out,
        N=N_int,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ============================================================================
# PATTERN AND REPLACEMENT
# ============================================================================

def pattern(x):
    """
    Match: torch.arange(0, end, device=dev) where the arange result is directly used
    Returns the arange result
    """
    from torch import device
    result = torch.arange(0, x, device=device(type='cuda', index=0))
    return result


def replacement_args(x):
    """
    Extract arguments needed for replacement.
    """
    from torch import device
    return (x, device(type='cuda', index=0), "arange")


def replacement_func():
    """
    Replacement function with routing.
    Route "arange": execute arange kernel
    """
    def dispatch_wrapper(x, dev, route="default"):
        if route == "arange":
            return triton_arange(x, dev)
        else:
            # Fallback - should not reach here
            raise ValueError(f"Unknown route: {route}")
    
    return dispatch_wrapper