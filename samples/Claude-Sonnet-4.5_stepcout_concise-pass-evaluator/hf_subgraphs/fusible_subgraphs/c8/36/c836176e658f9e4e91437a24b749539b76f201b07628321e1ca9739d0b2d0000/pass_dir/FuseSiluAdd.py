import torch
import triton
import triton.language as tl


# Pattern matching function - matches just the add operation
# The kernel will compute fused silu+add
def pattern(in_0, in_1):
    # Match simple add - framework will apply this to the silu+add pattern
    return in_0 + in_1


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel - since in_1 is already modified by inplace silu,
# we just need to compute in_1 + in_0
@triton.jit
def silu_add_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load in_0 (the tensor to add)
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    # Load in_1 (already contains silu result from inplace operation)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Just add - in_1 already has silu applied from the inplace operation
    out = in_1 + in_0
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_silu_add(in_0, in_1):
    """Fused silu + add kernel."""
    N = in_0.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(in_0)

    silu_add_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return triton_silu_add