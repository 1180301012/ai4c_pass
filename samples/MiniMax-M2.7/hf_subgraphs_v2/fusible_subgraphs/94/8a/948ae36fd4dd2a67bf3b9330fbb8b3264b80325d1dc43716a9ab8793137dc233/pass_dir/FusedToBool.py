import torch
import triton
import triton.language as tl

# ============================================================================
# TRITON KERNEL FOR TO_BOOL
# ============================================================================

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
# WRAPPER FUNCTION
# ============================================================================

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

def pattern(x):
    """
    Match: tensor.to(dtype=torch.bool)
    Returns the converted tensor
    """
    from torch import device
    result = x.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return result


def replacement_args(x):
    """
    Extract arguments needed for replacement.
    """
    return (x, "to_bool")


def replacement_func():
    """
    Replacement function with routing.
    Route "to_bool": execute bool conversion kernel
    """
    def dispatch_wrapper(x, route="default"):
        if route == "to_bool":
            return triton_to_bool(x)
        else:
            # Fallback - should not reach here
            raise ValueError(f"Unknown route: {route}")
    
    return dispatch_wrapper