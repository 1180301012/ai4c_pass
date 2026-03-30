import torch
import triton
import triton.language as tl

def pattern(tmp_10, tmp_11):
    """
    Fuse the subtract + masked_fill operations for better performance.
    Pattern matches: subtract, != comparison, masked_fill with -1000.0, == 0 comparison, masked_fill with 0.0
    """
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return tmp_16

def replacement_args(tmp_10, tmp_11):
    return (tmp_10, tmp_11)

@triton.jit
def fused_subtract_mask_kernel(
    tmp_10_ptr,
    tmp_11_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fusion kernel that computes:
    out[i] = -1000.0 if tmp_10[i] != tmp_11[i] else 0.0
    """
    # Each program processes a contiguous block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    tmp_10_val = tl.load(tmp_10_ptr + offsets, mask=mask, other=0.0)
    tmp_11_val = tl.load(tmp_11_ptr + offsets, mask=mask, other=0.0)
    
    # Compute difference
    diff = tmp_10_val - tmp_11_val
    
    # Apply optimized logic: if different, output -1000.0, otherwise 0.0
    # This avoids multiple memory operations and intermediate tensors
    result = tl.where(diff != 0, -1000.0, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_subtract_mask(tmp_10, tmp_11):
    """
    Wrapper function to launch the fused subtract+maskkernel
    """
    # Determine output shape and dtype
    output_shape = tmp_10.shape
    # Use float32 for output to match the masked_fill behavior
    out = torch.empty(output_shape, dtype=torch.float32, device=tmp_10.device)
    
    N = tmp_10.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_subtract_mask_kernel[(num_programs,)](
        tmp_10_ptr=tmp_10,
        tmp_11_ptr=tmp_11,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_subtract_mask