import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation graph
def pattern(in_0):
    """
    Match the computation pattern:
    1. Convert to float32
    2. Compute 1.0 - x
    3. Convert to bool
    4. masked_fill with -inf
    5. Multiply by the 1-x value
    """
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return tmp_4

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel with fixed configuration for minimal overhead
@triton.jit
def fused_invert_mask_multiply_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel with contiguous memory access pattern.
    Optimized for minimal kernel launch overhead.
    """
    # Calculate program start and process a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all elements in the block at once
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute 1.0 - x
    inv_x = 1.0 - x
    
    # Apply masking: -inf where inv_x != 0, else 0
    is_nonzero = tl.where(inv_x != 0.0, -float('inf'), 0.0)
    result = is_nonzero * inv_x
    
    # Store results
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_invert_mask_multiply(in_0):
    """
    Wrapper for the fused kernel.
    Optimized for minimal overhead with fixed block size.
    """
    n_elements = in_0.numel()
    
    # Get input shape and device
    shape = in_0.shape
    device = in_0.device
    
    # Allocate output with float32 dtype explicitly
    out = torch.empty(shape, dtype=torch.float32, device=device)
    
    # Use fixed block size that works well for small tensors
    BLOCK_SIZE = 512
    
    # Single program grid with num_warps=1 for minimal overhead
    fused_invert_mask_multiply_kernel[(1,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    
    return out

def replacement_func():
    return fused_invert_mask_multiply