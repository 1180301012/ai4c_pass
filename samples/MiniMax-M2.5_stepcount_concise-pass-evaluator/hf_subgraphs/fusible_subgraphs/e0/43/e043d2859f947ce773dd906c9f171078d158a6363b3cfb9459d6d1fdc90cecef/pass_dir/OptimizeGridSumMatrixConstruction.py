import torch
import triton
import triton.language as tl

# Pattern to match graph 1 (microsoft_beit-base-patch16-384):
# - Create coordinate grid 24x24
# - Element-wise operations (add 23, multiply 47)
# - Sum reduction
# - Matrix construction (577x577) with slice assignments
# - Return concatenated input and flattened matrix

def pattern(in_0, in_1):
    # Step 1: Concatenate inputs
    tmp_0 = torch.cat([in_1, in_0])
    
    # Step 2: Create coordinate grids (24x24)
    tmp_1 = torch.arange(24)
    tmp_2 = torch.arange(24)
    tmp_3 = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_1 = tmp_2 = None
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_3 = None
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_4 = tmp_5 = None
    tmp_7 = torch.flatten(tmp_6, 1)
    tmp_6 = None
    
    # Step 3: Indexing to create broadcastable shapes
    tmp_8 = tmp_7[slice(None, None, None), slice(None, None, None), None]
    tmp_9 = tmp_7[slice(None, None, None), None, slice(None, None, None)]
    tmp_7 = None
    
    # Step 4: Element-wise subtraction
    tmp_10 = tmp_8 - tmp_9
    tmp_8 = tmp_9 = None
    
    # Step 5: Permute and make contiguous
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_10 = None
    tmp_12 = tmp_11.contiguous()
    tmp_11 = None
    
    # Step 6: First element-wise op: add 23
    tmp_13 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_13 += 23
    tmp_14 = tmp_13
    tmp_13 = None
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_14
    tmp_15 = tmp_12
    tmp_14 = tmp_15 = None
    
    # Step 7: Second element-wise op: add 23
    tmp_16 = tmp_12[slice(None, None, None), slice(None, None, None), 1]
    tmp_16 += 23
    tmp_17 = tmp_16
    tmp_16 = None
    tmp_12[slice(None, None, None), slice(None, None, None), 1] = tmp_17
    tmp_18 = tmp_12
    tmp_17 = tmp_18 = None
    
    # Step 8: Third element-wise op: multiply by 47
    tmp_19 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_19 *= 47
    tmp_20 = tmp_19
    tmp_19 = None
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_20
    tmp_21 = tmp_12
    tmp_20 = tmp_21 = None
    
    # Step 9: Create zeros matrix and compute sum
    tmp_22 = torch.zeros(size=(577, 577), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_12 = None
    
    # Step 10: Slice assignments
    tmp_22[slice(1, None, None), slice(1, None, None)] = tmp_23
    tmp_24 = tmp_22
    tmp_23 = tmp_24 = None
    tmp_22[0, slice(0, None, None)] = 2209
    tmp_25 = tmp_22
    tmp_25 = None
    tmp_22[slice(0, None, None), 0] = 2210
    tmp_26 = tmp_22
    tmp_26 = None
    tmp_22[0, 0] = 2211
    tmp_27 = tmp_22
    tmp_27 = None
    
    # Step 11: Return
    tmp_28 = tmp_22.view(-1)
    tmp_22 = None
    return (tmp_0, tmp_28)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_grid_sum_matrix_kernel(
    out_ptr,
    n,
    offset_val,
    mult_val,
    const_row,
    const_col,
    const_diag,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that computes the grid->sum->matrix pattern directly.
    
    The original pattern:
    - Creates n x n coordinate grid
    - Computes (i - j) for all pairs using broadcasting
    - Channel 0: += offset, then *= mult
    - Channel 1: += offset
    - Sum across channels: (mult + 1) * (i - j + offset)
    - Embed into (n*n+1) x (n*n+1) matrix with constants
    
    For graph 1: n=24, offset=23, mult=47, size=577
    For graph 2: n=32, offset=31, mult=63, size=1025
    """
    # Each program handles one row of the output matrix (0 to n)
    pid = tl.program_id(0)
    row_idx = pid
    
    # Output is (n+1) x (n+1)
    output_size = n + 1
    
    if row_idx == 0:
        # First row: all const_row except first element is const_diag
        # out[0, 0] = const_diag
        # out[0, j] = const_row for j > 0
        col_offsets = tl.arange(0, BLOCK_SIZE)
        
        # Store diagonal element
        tl.store(out_ptr, const_diag)
        
        # Store rest of first row
        for j in range(1, BLOCK_SIZE):
            idx = j
            if idx < output_size:
                tl.store(out_ptr + idx, const_row)
                
    elif row_idx <= n:
        # Row r (1 to n): 
        # First column element is const_col
        # Elements 1 to n are computed values
        
        # Store first column value at [r, 0]
        tl.store(out_ptr + row_idx * output_size, const_col)
        
        # For columns 1 to n, compute the value
        # y = row_idx - 1 is the row coordinate (0 to n-1)
        y = row_idx - 1
        
        # mult_plus_1 = mult + 1
        mult_plus_1 = mult_val + 1
        
        for j in range(BLOCK_SIZE):
            idx = j
            if idx > 0 and idx <= n:
                x = idx - 1  # column coordinate (0 to n-1)
                
                # Compute the value: (mult + 1) * ((x - y) + offset)
                # The original computes i - j (signed), not |i - j|
                diff = x - y
                    
                value = mult_plus_1 * (diff + offset_val)
                tl.store(out_ptr + row_idx * output_size + idx, value)


@torch.fx.wrap
def optimized_grid_matrix_wrapper(in_0, in_1):
    """
    Wrapper function that launches the optimized Triton kernel.
    
    For graph 1: n=24, offset=23, mult=47, size=577
    const_row=2209, const_col=2210, const_diag=2211
    """
    # Parameters for graph 1 (base model)
    n = 24
    offset_val = 23
    mult_val = 47
    output_size = 577
    const_row = 2209
    const_col = 2210
    const_diag = 2211
    
    # Create output tensor
    out = torch.empty((output_size, output_size), dtype=torch.int64, device='cuda')
    
    # Configure block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with grid = output_size rows
    grid = (output_size,)
    
    fused_grid_sum_matrix_kernel[grid](
        out,
        n,
        offset_val,
        mult_val,
        const_row,
        const_col,
        const_diag,
        BLOCK_SIZE,
    )
    
    # Compute concatenated input
    tmp_0 = torch.cat([in_1, in_0])
    
    # Flatten the matrix for output
    tmp_28 = out.view(-1)
    
    return (tmp_0, tmp_28)


def replacement_func():
    return optimized_grid_matrix_wrapper