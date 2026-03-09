import torch
import triton
import triton.language as tl

def pattern(x):
    # tmp_1 = tmp_0.ne(1)
    tmp_1 = x.ne(1)
    # tmp_2 = tmp_1.int()
    tmp_2 = tmp_1.int()
    # tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    # tmp_4 = tmp_3.type_as(tmp_2)
    tmp_4 = tmp_3.type_as(tmp_2)
    # tmp_5 = tmp_4 + 0
    tmp_5 = tmp_4 + 0
    # tmp_6 = tmp_5 * tmp_2
    tmp_6 = tmp_5 * tmp_2
    # tmp_7 = tmp_6.long()
    tmp_7 = tmp_6.long()
    # tmp_8 = tmp_7 + 1
    tmp_8 = tmp_7 + 1
    return tmp_8

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_cumsum_mask_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Load entire row (assuming BLOCK_SIZE_N covers the full width)
    col_offset = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offset < n_cols
    
    # Load input data for the row
    x = tl.load(x_ptr + row_idx * n_cols + col_offset, mask=col_mask, other=0)
    
    # Step 1: Create boolean mask for elements != 1 and convert to int
    mask_ne_1 = (x != 1)
    mask_int = mask_ne_1.to(tl.int32)
    
    # Step 2: Compute cumulative sum along columns
    cumsum = tl.cumsum(mask_int * col_mask, axis=0)
    
    # Step 3: Apply mask (zero out positions where x == 1) 
    # This corresponds to tmp_6 = tmp_5 * tmp_2 in the original
    masked_cumsum = cumsum * mask_int
    
    # Step 4: Convert to int64 and add 1 (equivalent to tmp_7 + 1)
    result = masked_cumsum.to(tl.int64) + 1
    
    # Final result: 0 where original element was 1, cumsum+1 otherwise
    final_result = tl.where(mask_ne_1, result, 0)
    
    # Store result
    tl.store(out_ptr + row_idx * n_cols + col_offset, final_result, mask=col_mask)

@torch.fx.wrap
def optimized_cumsum_mask(x):
    n_rows, n_cols = x.shape
    
    # Choose optimal block sizes
    BLOCK_SIZE_N = min(1024, n_cols)  # Handle up to 1024 columns per row
    BLOCK_SIZE_M = 1  # One row per program
    
    # Calculate grid dimensions
    n_grid = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.int64)
    
    # Launch kernel - one program per row
    optimized_cumsum_mask_kernel[(n_rows, n_grid)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return optimized_cumsum_mask