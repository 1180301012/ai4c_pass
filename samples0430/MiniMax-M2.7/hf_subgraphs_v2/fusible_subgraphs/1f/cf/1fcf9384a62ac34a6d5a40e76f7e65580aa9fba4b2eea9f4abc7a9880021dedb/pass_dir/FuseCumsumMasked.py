import torch
import triton
import triton.language as tl


@triton.jit
def fused_cumsum_masked_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel for the sequence:
    1. tmp_1 = in_0 != 1  (boolean mask)
    2. tmp_2 = tmp_1.int()  (0/1 values)
    3. tmp_3 = cumsum(tmp_2, dim=1)  (cumulative sum along rows)
    4. tmp_4 = tmp_3.type_as(tmp_2)  (type conversion - no-op)
    5. tmp_5 = tmp_4 + 0  (no-op)
    6. tmp_6 = tmp_5 * tmp_2  (masked selection)
    7. tmp_7 = tmp_6.long()  (convert to long)
    8. tmp_8 = tmp_7 + 1  (offset by 1)
    
    Strategy: One thread per row, vectorized processing
    """
    # Each block handles one row
    row_idx = tl.program_id(0)
    
    # Row starting offset
    row_start = row_idx * n_cols
    
    # Calculate cumsum incrementally as we scan through the row
    # Use int64 to match output type
    cumsum_val = tl.cast(0, tl.int64)
    
    # Process in blocks for better cache utilization and memory coalescing
    for block_start in range(0, n_cols, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n_cols)
        block_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_offsets < n_cols
        
        # Load input values for this block
        input_offsets = row_start + block_offsets
        mask_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0)
        
        # tmp_1: in_0 != 1 (vectorized)
        is_not_one = mask_vals != 1
        
        # tmp_2: Convert to int (0 or 1) - vectorized
        mask_int = tl.cast(is_not_one, tl.int64)
        
        # Compute prefix sums for this block relative to cumsum start
        block_prefix = tl.cumsum(mask_int, axis=0)
        
        # Compute result: cumsum * mask + 1
        # If mask=1: result = (cumsum_start + block_prefix) * 1 + 1 = cumsum_start + block_prefix + 1
        # If mask=0: result = (cumsum_start + block_prefix) * 0 + 1 = 1
        masked_cumsum = (cumsum_val + block_prefix) * mask_int
        result_block = masked_cumsum + 1
        
        # Store results for this block
        output_offsets = row_start + block_offsets
        tl.store(output_ptr + output_offsets, result_block, mask=mask)
        
        # Update cumsum for next block - cumsum_val at end of this block
        # cumsum_val after processing block = sum of all mask values in block
        cumsum_val = cumsum_val + tl.sum(mask_int)


@torch.fx.wrap
def fused_cumsum_masked_wrapper(input_tensor):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Args:
        input_tensor: Input tensor of shape [n_rows, n_cols], dtype torch.int64
        
    Returns:
        Output tensor of shape [n_rows, n_cols], dtype torch.int64
    """
    n_rows, n_cols = input_tensor.shape
    n_elements = n_rows * n_cols
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor)
    
    # Define block size - use power of 2 for alignment
    BLOCK_SIZE = 64
    
    # Launch grid: one block per row
    grid = (n_rows,)
    
    fused_cumsum_masked_kernel[grid](
        input_tensor,
        output,
        n_elements,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0):
    """
    Pattern matching function for the fused computation.
    Matches the exact computation graph:
    - in_0.ne(1) -> int -> cumsum -> type_as -> +0 -> *mask -> long -> +1
    """
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_cumsum_masked_wrapper