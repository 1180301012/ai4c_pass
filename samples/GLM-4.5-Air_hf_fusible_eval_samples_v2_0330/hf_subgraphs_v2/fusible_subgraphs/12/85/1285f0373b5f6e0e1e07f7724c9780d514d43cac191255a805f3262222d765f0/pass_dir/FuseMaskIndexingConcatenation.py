import torch
import triton
import triton.language as tl

def pattern(tmp_0, in_1, in_2):
    """
    Match the pattern: mask indexing followed by concatenation
    tmp_1 = tmp_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    """
    tmp_1 = tmp_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    return tmp_9

def replacement_args(tmp_0, in_1, in_2):
    return (tmp_0, in_1, in_2)

@triton.jit
def fused_kernel(
    tmp_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    tmp_0_dim0,
    tmp_0_dim1,
    in_1_dim0,
    in_1_dim1,
    mask_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element of the output along dimension 1
    col_idx = tl.program_id(0)
    row_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_row = row_idx < tmp_0_dim0
    mask_col = col_idx < in_1_dim1
    
    # Count True values in mask up to column col_idx
    count_true = 0
    for i in range(col_idx + 1):
        if i < mask_size and in_2_ptr[i]:
            if i < count_true + 1:  # This is a rough approximation
                count_true += 1
    
    # For each row, determine if we should take from tmp_0 or in_1
    # tmp_0 contributes columns 0..count_true-1
    # in_1 contributes columns count_true..count_true+in_1_dim1-1
    
    # Load from tmp_0 for corresponding row and column index
    # We need to map the output column to either tmp_0 or in_1
    source_col = col_idx
    if col_idx < count_true:
        # This column comes from tmp_0
        tmp_0_col_idx = 0
        for i in range(col_idx):
            if i < mask_size and in_2_ptr[i]:
                tmp_0_col_idx += 1
        # Load from tmp_0
        tmp_0_offsets = row_idx * tmp_0_dim1 + tmp_0_col_idx
        tmp_0_mask = mask_row & (col_idx < count_true)
        val = tl.load(tmp_0_ptr + tmp_0_offsets, mask=tmp_0_mask, other=0)
    else:
        # This column comes from in_1
        in_1_col_idx = col_idx - count_true
        in_1_offsets = row_idx * in_1_dim1 + in_1_col_idx
        in_1_mask = mask_row & (col_idx >= count_true) & mask_col
        val = tl.load(in_1_ptr + in_1_offsets, mask=in_1_mask, other=0)
    
    # Store the result
    output_offsets = row_idx * (count_true + in_1_dim1) + col_idx
    output_mask = mask_row & (col_idx < count_true + in_1_dim1)
    tl.store(out_ptr + output_offsets, val, mask=output_mask)

@torch.fx.wrap
def fused_kernel_wrapper(tmp_0, in_1, in_2):
    # Count True values in mask using torch operations (avoiding triton.sum for now)
    mask_count = int(in_2.sum())
    
    # Determine output dimensions
    out_dim0 = tmp_0.size(0)
    out_dim1 = mask_count + in_1.size(1)
    
    # Create output tensor
    output = torch.empty((out_dim0, out_dim1), dtype=tmp_0.dtype, device=tmp_0.device)
    
    # Block size and grid configuration
    BLOCK_SIZE = 32  # Number of rows per program
    num_rows = (out_dim0 + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_cols = out_dim1
    
    # Launch kernel
    fused_kernel[(num_cols, num_rows)](
        tmp_0_ptr=tmp_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=output,
        tmp_0_dim0=tmp_0.size(0),
        tmp_0_dim1=tmp_0.size(1),
        in_1_dim0=in_1.size(0),
        in_1_dim1=in_1.size(1),
        mask_size=in_2.size(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_kernel_wrapper