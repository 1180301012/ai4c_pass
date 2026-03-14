import torch
import triton
import triton.language as tl


# Pattern matching function - matches the cat + normalize pattern
def pattern(x):
    # Match: torch.cat([x], 1) -> normalize(..., dim=1)
    t = torch.cat([x], 1)
    out = torch.nn.functional.normalize(t, p=2, dim=1)
    return out


# Argument extraction function
def replacement_args(x):
    return (x,)


# Optimized Triton kernel for L2 normalization along dim=1
@triton.jit
def normalize_kernel(
    input_ptr,
    output_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Check if row is valid
    if row_idx >= rows:
        return
    
    # Calculate row offset
    row_offset = row_idx * cols
    
    # Load the entire row and compute L2 norm
    # Process in blocks to fit in SRAM
    sum_squares = 0.0
    
    # Compute sum of squares
    for col_block in range(0, cols, BLOCK_SIZE):
        col_offsets = col_block + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < cols
        
        # Load values
        values = tl.load(input_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        
        # Accumulate squares
        sum_squares += tl.sum(values * values, axis=0)
    
    # Compute norm
    norm = tl.sqrt(sum_squares + 1e-8)  # add epsilon for numerical stability
    
    # Normalize and store
    for col_block in range(0, cols, BLOCK_SIZE):
        col_offsets = col_block + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < cols
        
        # Load values
        values = tl.load(input_ptr + row_offset + col_offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = values / norm
        
        # Store
        tl.store(output_ptr + row_offset + col_offsets, normalized, mask=mask)


@torch.fx.wrap
def triton_normalize(x):
    # Get dimensions
    rows, cols = x.shape
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Choose block size based on columns (768 typical)
    BLOCK_SIZE = 768
    
    # Launch kernel - one program per row
    grid = (rows,)
    
    normalize_kernel[grid](
        x,
        output,
        rows,
        cols,
        BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return triton_normalize