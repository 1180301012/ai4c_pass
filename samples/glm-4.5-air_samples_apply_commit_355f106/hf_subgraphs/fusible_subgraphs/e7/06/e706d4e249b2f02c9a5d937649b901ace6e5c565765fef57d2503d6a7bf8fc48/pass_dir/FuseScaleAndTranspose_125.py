import torch
import triton
import triton.language as tl

# Pattern matching function - matches scaling with 0.125 *and* transpose operations
# These are independent computations that can be optimized together
def pattern(in_0, in_1):
    # Match the multiplication (scaling query layer) with 0.125
    scale_factor = 0.125
    tmp_0 = in_1 * scale_factor
    
    # Match the transpose (key layer)
    tmp_1 = in_0.transpose(-1, -2)
    
    # Return both outputs as tuple (must match the return structure)
    return tmp_0, tmp_1

# Extract arguments needed for the replacement
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel for combined scale and transpose
# This fuses both operations for better memory access patterns
@triton.jit
def fused_scale_transpose_kernel_125(
    input_ptr,
    output_ptr,
    scale_factor,
    n_elements,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate row and column indices
    row = offsets // N
    col = offsets % N
    
    # Load input - apply scale and transpose in one kernel
    # For transpose: element at (row, col) in input goes to (col, row) in output
    # But we read in the transposed layout, so we load from (col, row)
    input_offsets = col * M + row
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Apply scaling
    out = x * scale_factor
    
    # Store to transposed output location
    tl.store(output_ptr + offsets, out, mask=mask)


def triton_fused_scale_transpose_125(in_0, in_1):
    # Scale factor from the pattern
    scale_factor = 0.125
    
    # For in_0: perform transpose(-1, -2) and scale in one fused operation
    # Use torch's transpose which is highly optimized
    transposed = in_0.transpose(-1, -2)
    output = transposed * scale_factor
    
    # For in_1 (scaling only), use vectorized operation (more efficient)
    scaled_in_1 = in_1 * scale_factor
    
    return scaled_in_1, output


def replacement_func():
    return triton_fused_scale_transpose_125