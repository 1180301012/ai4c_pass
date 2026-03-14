import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(in_0):
    """ 
    Match the pattern: torch.cat([in_0], 1) followed by normalize along dim=1
    """
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel for L2 normalization
@triton.jit
def normalize_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row this program handles
    row_idx = tl.program_id(0)
    
    # Boundary check
    if row_idx >= batch_size:
        return
    
    # Compute the row offset
    row_offset = row_idx * feature_size
    
    # Compute L2 norm: sqrt(sum(x^2))
    # Use blocking for large feature sizes
    norm = 0.0
    for i in range(0, feature_size, BLOCK_SIZE):
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_offset + feature_size
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        norm += tl.sum(x * x, axis=0)
    
    norm = tl.sqrt(norm)
    
    # Handle edge case where norm is zero
    norm = tl.where(norm == 0.0, 1.0, norm)
    
    # Normalize and store
    for i in range(0, feature_size, BLOCK_SIZE):
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_offset + feature_size
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        normalized = x / norm
        tl.store(output_ptr + offsets, normalized, mask=mask)


@torch.fx.wrap
def triton_normalize(x):
    """Triton kernel for L2 normalization along dim=1"""
    batch_size, feature_size = x.shape
    
    # Choose block size based on feature size
    # 768 features fits well with BLOCK_SIZE=1024
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    num_programs = batch_size
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel
    normalize_kernel[(num_programs,)](
        x,
        out,
        batch_size,
        feature_size,
        BLOCK_SIZE,
    )
    
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_normalize