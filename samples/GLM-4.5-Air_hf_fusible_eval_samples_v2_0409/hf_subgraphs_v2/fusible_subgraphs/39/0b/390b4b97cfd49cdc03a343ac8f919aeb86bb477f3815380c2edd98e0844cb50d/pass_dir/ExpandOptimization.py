import torch
import triton
import triton.language as tl

def pattern(x):
    return x.expand(3, -1, -1)

def replacement_args(x):
    return (x,)

@triton.jit
def expand_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a specific element in the expanded tensor
    pid = tl.program_id(0)
    
    # Calculate position in the expanded tensor (3, batch_size, seq_len)
    row = pid // (batch_size * seq_len)
    batch_idx = (pid % (batch_size * seq_len)) // seq_len
    col = (pid % (batch_size * seq_len)) % seq_len
    
    # Calculate source position in input tensor (1, batch_size, seq_len)
    src_offset = batch_idx * seq_len + col
    
    # Calculate destination position in output tensor (3, batch_size, seq_len)
    dst_offset = row * batch_size * seq_len + batch_idx * seq_len + col
    
    # Load from input tensor (source is always row 0)
    src_x_offset = src_offset
    x_val = tl.load(x_ptr + src_x_offset)
    
    # Store to output tensor (all 3 rows)
    tl.store(out_ptr + dst_offset, x_val)

@torch.fx.wrap
def optimized_expand(x):
    # Handle both 1D and 2D input tensors
    if x.ndim == 1:
        batch_size, = x.shape
        seq_len = 1  # For 1D, treat as (batch_size, 1)
    elif x.ndim == 2:
        batch_size, seq_len = x.shape
    else:
        # For higher dimensions, use the last two dimensions
        batch_size = x.shape[-2]
        seq_len = x.shape[-1]
    
    out = torch.empty((3, batch_size, seq_len), dtype=x.dtype, device=x.device)
    
    # Calculate total elements and tile size
    total_elements = 3 * batch_size * seq_len
    if total_elements > 65536:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 256
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_programs,)
    expand_kernel[grid](
        x,
        out,
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_expand