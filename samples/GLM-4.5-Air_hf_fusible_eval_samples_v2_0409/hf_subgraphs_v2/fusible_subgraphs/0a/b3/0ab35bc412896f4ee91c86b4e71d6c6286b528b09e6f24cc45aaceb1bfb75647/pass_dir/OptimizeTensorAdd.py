import torch
import triton
import triton.language as tl

# Pattern matching function for tensor addition
def pattern(x, y):
    return x + y

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized addition kernel
@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate total elements
    total_elements = batch_size * seq_len * hidden_dim
    
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load tensors with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_tensor_add(x, y):
    batch_size, seq_len, hidden_dim = x.shape
    total_elements = batch_size * seq_len * hidden_dim
    
    # For small tensors, use native torch addition instead of Triton
    # because the overhead of kernel launch might outweigh the benefits
    if total_elements <= 2048:  # Small tensor threshold
        return x + y
    
    # For larger tensors, use Triton optimized version
    # Choose optimal block size - multiple of 256 for good GPU occupancy
    BLOCK_SIZE = min(1024, total_elements)
    
    # Calculate number of programs needed
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    triton_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_tensor_add