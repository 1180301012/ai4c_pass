import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_1, in_0):
    relu_out = torch.nn.functional.relu(in_1)
    flat_out = torch.flatten(relu_out, 2)
    norm_out = torch.functional.norm(flat_out, dim=-1, keepdim=True)
    scaled_norm = norm_out * 0.14433756729740643
    clamped_norm = scaled_norm.clamp(min=1e-05)
    div_out = flat_out / clamped_norm
    result = div_out * in_0
    return result

# Argument extraction function

def replacement_args(in_1, in_0):
    # Return all arguments needed for replacement
    scale_val = 0.14433756729740643
    clamp_val = 1e-05
    return (in_1, in_0, scale_val, clamp_val)

# Triton kernel for fused normalization
@triton.jit
def fused_normalize_kernel(
    in_ptr,
    scale_val,
    clamp_val,
    out_ptr,
    batch_size,
    seq_len,
    d,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one [batch, seq] row and a portion of the last dimension
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Calculate offset for this row
    row_offset = (batch_idx * seq_len + seq_idx) * d

    # Compute squared sum of the row
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, d, BLOCK_SIZE):
        # Load a block of data
        x = tl.load(in_ptr + row_offset + i, 
                   mask=i + tl.arange(0, BLOCK_SIZE) < d,
                   other=0.0)
        # Square and accumulate
        sum_sq += tl.sum(x * x)
    
    # Compute the norm and clamp
    norm = tl.sqrt(sum_sq) * scale_val
    clamped_norm = tl.maximum(norm, clamp_val)

    # Normalize and write the result
    for i in range(0, d, BLOCK_SIZE):
        # Load input data
        x = tl.load(in_ptr + row_offset + i, 
                   mask=i + tl.arange(0, BLOCK_SIZE) < d,
                   other=0.0)
        # Normalize
        y = x / clamped_norm
        # Store result
        tl.store(out_ptr + row_offset + i, y,
                mask=i + tl.arange(0, BLOCK_SIZE) < d)

# Kernel wrapper with memory handling
@torch.fx.wrap
def fused_normalize(x, scale_val, clamp_val, in_0):
    # Get tensor dimensions after flatten
    batch, seq, d = x.shape
    
    # Calculate grid size (one thread block per [batch, seq] row)
    BLOCK_SIZE = 64  # Optimized for memory coalescing
    grid_size = (batch * seq + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch kernel
    fused_normalize_kernel[grid_size,](
        x, 
        scale_val, 
        clamp_val, 
        out, 
        batch, 
        seq, 
        d, 
        BLOCK_SIZE
    )

    # Final scaling with in_0
    return out * in_0

# Replacement function returns the kernel wrapper

def replacement_func():
    return fused_normalize