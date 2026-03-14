import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern to match the entire forward function.
    """
    tmp_0 = torch.cat([in_1, in_0])
    tmp_1 = torch.arange(24)
    tmp_2 = torch.arange(24)
    tmp_3 = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)
    tmp_8 = tmp_7[slice(None, None, None), slice(None, None, None), None]
    tmp_9 = tmp_7[slice(None, None, None), None, slice(None, None, None)]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()
    tmp_13 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_13 += 23
    tmp_14 = tmp_13
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_14
    tmp_15 = tmp_12
    tmp_16 = tmp_12[slice(None, None, None), slice(None, None, None), 1]
    tmp_16 += 23
    tmp_17 = tmp_16
    tmp_12[slice(None, None, None), slice(None, None, None), 1] = tmp_17
    tmp_18 = tmp_12
    tmp_19 = tmp_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_19 *= 47
    tmp_20 = tmp_19
    tmp_12[slice(None, None, None), slice(None, None, None), 0] = tmp_20
    tmp_21 = tmp_12
    tmp_22 = torch.zeros(size=(577, 577), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[slice(1, None, None), slice(1, None, None)] = tmp_23
    tmp_24 = tmp_22
    tmp_22[0, slice(0, None, None)] = 2209
    tmp_25 = tmp_22
    tmp_22[slice(0, None, None), 0] = 2210
    tmp_26 = tmp_22
    tmp_22[0, 0] = 2211
    tmp_27 = tmp_22
    tmp_28 = tmp_22.view(-1)
    return (tmp_0, tmp_28)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def relative_position_bias_kernel(
    output_ptr,
    grid_size: tl.constexpr,
    output_size: tl.constexpr,
    offset: tl.constexpr,
    multiplier: tl.constexpr,
    special_val_0: tl.constexpr,
    special_val_1: tl.constexpr,
    special_val_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel to compute relative position bias indices.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Compute 2D indices from linear offset
    row = offsets // output_size
    col = offsets % output_size
    
    # Mask for valid indices
    mask = offsets < (output_size * output_size)
    
    # Initialize result with special values
    result = tl.full((BLOCK_SIZE,), special_val_2, dtype=tl.int64)
    
    # Case 1: Both row and col are > 0 (inner grid)
    inner_mask = (row >= 1) & (col >= 1) & mask
    
    # Map to grid coordinates (row-1, col-1)
    grid_row = row - 1
    grid_col = col - 1
    
    pos_i = grid_row
    pos_j = grid_col
    
    # Each position in grid_size^2 corresponds to (y, x) coordinates in grid_size x grid_size
    i_y = pos_i // grid_size
    i_x = pos_i % grid_size
    j_y = pos_j // grid_size
    j_x = pos_j % grid_size
    
    # Relative position
    rel_y = i_y - j_y + offset
    rel_x = i_x - j_x + offset
    
    # Combined index
    rel_idx = rel_y * multiplier + rel_x
    
    # Apply inner mask
    result = tl.where(inner_mask, rel_idx, result)
    
    # Case 2: row == 0, col != 0 (first row except [0,0])
    first_row_mask = (row == 0) & (col > 0) & mask
    result = tl.where(first_row_mask, special_val_0, result)
    
    # Case 3: col == 0, row != 0 (first column except [0,0])
    first_col_mask = (col == 0) & (row > 0) & mask
    result = tl.where(first_col_mask, special_val_1, result)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def cat_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    in_0_size,
    in_1_size,
    dim1_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to concatenate two tensors along dimension 0.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = (in_0_size + in_1_size) * dim1_size
    mask = offsets < total_elements
    
    row = offsets // dim1_size
    col = offsets % dim1_size
    
    # If row < in_1_size, read from in_1, else read from in_0
    in_1_mask = (row < in_1_size) & mask
    in_0_mask = (row >= in_1_size) & mask
    
    # Read from in_1
    in_1_idx = row * dim1_size + col
    val_1 = tl.load(in_1_ptr + in_1_idx, mask=in_1_mask, other=0.0)
    
    # Read from in_0
    in_0_row = row - in_1_size
    in_0_idx = in_0_row * dim1_size + col
    val_0 = tl.load(in_0_ptr + in_0_idx, mask=in_0_mask, other=0.0)
    
    # Combine
    result = tl.where(in_1_mask, val_1, val_0)
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_forward_24(in_0, in_1):
    """
    Optimized forward function for 24x24 grid.
    """
    # Concatenate inputs
    in_0_shape = in_0.shape
    in_1_shape = in_1.shape
    out_shape = (in_0_shape[0] + in_1_shape[0], in_0_shape[1])
    tmp_0 = torch.empty(out_shape, dtype=in_0.dtype, device='cuda')
    
    BLOCK_SIZE = 256
    total_elements = out_shape[0] * out_shape[1]
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    cat_kernel[(num_programs,)](
        in_0,
        in_1,
        tmp_0,
        in_0_shape[0],
        in_1_shape[0],
        in_0_shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Compute relative position bias
    grid_size = 24
    output_size = 577
    offset = 23
    multiplier = 47
    special_val_0 = 2209
    special_val_1 = 2210
    special_val_2 = 2211
    
    total_elements = output_size * output_size
    tmp_28 = torch.empty(total_elements, dtype=torch.int64, device='cuda')
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    relative_position_bias_kernel[(num_programs,)](
        tmp_28,
        grid_size=grid_size,
        output_size=output_size,
        offset=offset,
        multiplier=multiplier,
        special_val_0=special_val_0,
        special_val_1=special_val_1,
        special_val_2=special_val_2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (tmp_0, tmp_28)


def replacement_func():
    return fused_forward_24