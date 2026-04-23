import torch
import triton
import triton.language as tl

# ============================================================================
# TRITON KERNELS
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


@triton.jit
def to_bool_kernel(
    in_ptr,
    out_ptr,
    total_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to convert int64 tensor to bool"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    # Load int64 value and convert to bool (non-zero = True)
    val = tl.load(in_ptr + offsets, mask=mask, other=0)
    bool_val = val != 0
    tl.store(out_ptr + offsets, bool_val, mask=mask)


# ============================================================================
# WRAPPER FUNCTIONS
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


@torch.fx.wrap
def triton_to_bool(input_tensor):
    """Convert int64 tensor to bool using Triton kernel"""
    total = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    if num_programs < 1:
        num_programs = 1
    
    out = torch.empty_like(input_tensor, dtype=torch.bool)
    to_bool_kernel[(num_programs,)](
        in_ptr=input_tensor,
        out_ptr=out,
        total_elements=total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ============================================================================
# PATTERN AND REPLACEMENT
# ============================================================================

def pattern(in_0, end, dev):
    """
    Match: torch.arange(0, end, device=dev) + in_0.to(device=dev, dtype=torch.bool)
    Returns tuple (arange_result, bool_result)
    """
    arange_result = torch.arange(0, end, device=dev)
    bool_result = in_0.to(device=dev, dtype=torch.bool)
    return arange_result, bool_result


def replacement_args(in_0, end, dev):
    """
    Extract arguments needed for replacement.
    Route string "fused" indicates this is the fused implementation.
    """
    return (in_0, end, dev, "fused")


def replacement_func():
    """
    Shared replacement function with routing.
    Route "fused": execute both kernels (can be parallelized)
    """
    def dispatch_wrapper(in_0, end, dev, route="default"):
        if route == "fused":
            # Both operations are independent - can potentially be parallelized
            # For single-gpu, sequential execution is fine
            arange_result = triton_arange(end, dev)
            bool_result = triton_to_bool(in_0)
            return (arange_result, bool_result)
        else:
            # Fallback - should not reach here
            raise ValueError(f"Unknown route: {route}")
    
    return dispatch_wrapper