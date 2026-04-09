import torch
import triton
import triton.language as tl

# Simple transpose pattern to test mechanism
def pattern(x):
    """Simple transpose pattern"""
    return x.transpose(1, 2)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Simple transpose kernel
@triton.jit
def simple_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    features,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    feature_id = tl.program_id(2)
    
    # Calculate memory offset
    src_offset = batch_id * seq_len * features + seq_id * features + feature_id
    dst_offset = batch_id * features * seq_len + feature_id * seq_len + seq_id
    
    # Load and store with transpose
    mask = feature_id < features
    value = tl.load(x_ptr + src_offset, mask=mask, other=0.0)
    tl.store(out_ptr + dst_offset, value, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def simple_transpose(x):
    batch_size, seq_len, features = x.shape
    
    # Output tensor
    out = torch.empty(batch_size, features, seq_len, dtype=x.dtype, device=x.device)
    
    # Block size
    BLOCK_SIZE = 64
    
    # Calculate grid dimensions
    grid = (batch_size, seq_len, features)
    
    # Launch kernel
    simple_transpose_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        features=features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return simple_transpose