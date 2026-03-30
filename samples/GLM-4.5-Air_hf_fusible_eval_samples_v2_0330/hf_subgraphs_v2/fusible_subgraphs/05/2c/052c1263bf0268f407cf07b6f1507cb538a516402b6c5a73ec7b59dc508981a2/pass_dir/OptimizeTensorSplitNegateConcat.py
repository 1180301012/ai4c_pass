import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_4):
    # Split in_2 into two halves
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    # Negate second half
    tmp_3 = -tmp_2
    # Concatenate back together
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    # Element-wise multiply with in_4
    tmp_5 = tmp_4 * in_4
    # Original multiplication
    tmp_0 = in_2 * in_1
    # Add results
    tmp_6 = tmp_0 + tmp_5
    return tmp_6

def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)

@triton.jit
def split_negate_concat_kernel(
    in_2_ptr, in_1_ptr, in_4_ptr, out_ptr,
    n_rows, n_cols_128,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a row and a block of columns
    row_idx = tl.program_id(0)
    col_offset = tl.program_id(1) * BLOCK_SIZE
    
    # Calculate pointer offsets for this row
    row_size = n_cols_128 * 2
    in_2_ptr_row = in_2_ptr + row_idx * row_size
    in_1_ptr_row = in_1_ptr + row_idx * row_size  
    in_4_ptr_row = in_4_ptr + row_idx * row_size
    out_ptr_row = out_ptr + row_idx * row_size
    
    # Process first half (0-127)
    mask1 = (col_offset + tl.arange(0, BLOCK_SIZE)) < n_cols_128
    offsets1 = col_offset + tl.arange(0, BLOCK_SIZE)
    
    # Load first half data
    in_2_first = tl.load(in_2_ptr_row + offsets1, mask=mask1, other=0.0)
    in_1_first = tl.load(in_1_ptr_row + offsets1, mask=mask1, other=0.0)
    in_4_first = tl.load(in_4_ptr_row + offsets1, mask=mask1, other=0.0)
    
    # Load second half data (from original second half position, but use it for first half output)
    mask2 = (col_offset + tl.arange(0, BLOCK_SIZE)) < n_cols_128
    offsets2 = col_offset + tl.arange(0, BLOCK_SIZE)
    
    # Second half elements from original tensor
    in_2_second_orig = tl.load(in_2_ptr_row + n_cols_128 + offsets2, mask=mask2, other=0.0)
    in_1_second = tl.load(in_1_ptr_row + n_cols_128 + offsets2, mask=mask2, other=0.0)
    in_4_second = tl.load(in_4_ptr_row + n_cols_128 + offsets2, mask=mask2, other=0.0)
    
    # Compute results for both halves
    # For positions 0-127: output = in_2 * in_1 + in_4 * in_2  (first half unchanged)
    # For positions 128-255: output = in_2 * in_1 + in_4 * (-in_2_second_half)
    
    # First half output (positions 0-127): in_2 * in_1 + in_4 * in_2
    out_first = in_2_first * in_1_first + in_4_first * in_2_first
    
    # Second half output (positions 128-255): in_2 * in_1 + in_4 * (-in_2_from_first_half)
    # We need data from first half but placed in second half position
    first_half_data_for_second = tl.load(in_2_ptr_row + offsets2, mask=mask2, other=0.0)
    out_second = in_2_second_orig * in_1_second + in_4_second * (-first_half_data_for_second)
    
    # Store results
    tl.store(out_ptr_row + offsets1, out_first, mask=mask1)
    tl.store(out_ptr_row + n_cols_128 + offsets2, out_second, mask=mask2)

@torch.fx.wrap
def split_negate_concat_optimized(in_2, in_1, in_4):
    # Get tensor dimensions  
    n_rows = in_2.numel() // (256)  # Total elements divided by columns
    n_cols_128 = 128
    
    # Create output tensor
    out = torch.empty_like(in_2)
    
    # Use optimal block size for these tensors
    BLOCK_SIZE = 128
    
    # Calculate grid dimensions
    num_rows = n_rows
    num_blocks_cols = 1  # We process each row in one block
    
    # Launch kernel
    split_negate_concat_kernel[(num_rows, num_blocks_cols)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_4_ptr=in_4,
        out_ptr=out,
        n_rows=n_rows,
        n_cols_128=n_cols_128,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return split_negate_concat_optimized