import torch
import triton
import triton.language as tl

# Triton kernel for copying tensor elements (for cat operation)
@triton.jit
def copy_kernel(src_ptr, dst_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)

# Triton kernel for copying tensor elements with destination offset
@triton.jit
def copy_kernel_offset(src_ptr, dst_ptr, dst_offset, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, vals, mask=mask)

# Triton kernel for index table generation - directly computes each element
@triton.jit
def index_table_kernel(out_ptr, N, grid_size, two_n_minus_1, n_minus_1,
                       row_fill_val, col_fill_val, origin_fill_val, 
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = grid_size * grid_size
    mask = offsets < total
    
    row = offsets // grid_size
    col = offsets % grid_size
    
    is_first_row = row == 0
    is_first_col = col == 0
    is_origin = is_first_row & is_first_col
    
    # Safe computation for main grid - use maximum to ensure non-negative values
    r = tl.maximum(row - 1, 0)
    c = tl.maximum(col - 1, 0)
    
    # Compute meshgrid coordinates from flattened position
    i1 = r // N  # row coordinate in meshgrid
    j1 = r % N   # column coordinate in meshgrid  
    i2 = c // N  # row coordinate for second position
    j2 = c % N   # column coordinate for second position
    
    # Compute shifted differences
    i_diff = i1 - i2 + n_minus_1
    j_diff = j1 - j2 + n_minus_1
    
    # Compute final index value: (2N-1) * i_diff + j_diff
    main_val = two_n_minus_1 * i_diff + j_diff
    
    # Select correct value based on position
    val = tl.where(is_origin, origin_fill_val,
           tl.where(is_first_row, row_fill_val,
           tl.where(is_first_col, col_fill_val,
           main_val)))
    
    tl.store(out_ptr + offsets, val.to(tl.int64), mask=mask)

# Triton kernel for cat operation - copies two tensors into one concatenated tensor
@triton.jit
def cat_kernel(in1_ptr, in0_ptr, out_ptr, in1_numel, total_numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_numel
    
    from_in1 = offsets < in1_numel
    in0_offsets = offsets - in1_numel
    
    # Load from appropriate source
    in1_vals = tl.load(in1_ptr + offsets, mask=from_in1 & mask, other=0.0)
    in0_vals = tl.load(in0_ptr + in0_offsets, mask=(~from_in1) & mask, other=0.0)
    
    vals = tl.where(from_in1, in1_vals, in0_vals)
    tl.store(out_ptr + offsets, vals, mask=mask)

# Kernel wrapper for cat operation only
@torch.fx.wrap
def triton_cat(in_0, in_1):
    total_rows = in_1.shape[0] + in_0.shape[0]
    cols = in_0.shape[1]
    cat_out = torch.empty((total_rows, cols), dtype=in_0.dtype, device=in_0.device)
    
    in1_numel = in_1.numel()
    total_numel = cat_out.numel()
    BLOCK_SIZE = 1024
    num_programs = (total_numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    cat_kernel[(num_programs,)](in_1, in_0, cat_out, in1_numel, total_numel, BLOCK_SIZE=BLOCK_SIZE)
    
    return cat_out

# Kernel wrapper for index table computation  
@torch.fx.wrap
def triton_index_table(route_str):
    # Route-specific parameters
    if route_str == "n32":
        N = 32
        grid_size = 1025
        two_n_minus_1 = 63
        n_minus_1 = 31
        row_fill = 3969
        col_fill = 3970
        origin_fill = 3971
    elif route_str == "n14":
        N = 14
        grid_size = 197
        two_n_minus_1 = 27
        n_minus_1 = 13
        row_fill = 729
        col_fill = 730
        origin_fill = 731
    elif route_str == "n24":
        N = 24
        grid_size = 577
        two_n_minus_1 = 47
        n_minus_1 = 23
        row_fill = 2209
        col_fill = 2210
        origin_fill = 2211
    else:
        raise ValueError(f"Unknown route: {route_str}")
    
    table_out = torch.empty((grid_size * grid_size,), dtype=torch.int64, device=torch.cuda.current_device())
    BLOCK_SIZE_TABLE = 1024
    total_elements = grid_size * grid_size
    num_programs_table = (total_elements + BLOCK_SIZE_TABLE - 1) // BLOCK_SIZE_TABLE
    index_table_kernel[(num_programs_table,)](
        table_out, N, grid_size, two_n_minus_1, n_minus_1,
        row_fill, col_fill, origin_fill, BLOCK_SIZE=BLOCK_SIZE_TABLE
    )
    
    return table_out

# The replacement_func returns the same function object for all passes
def replacement_func():
    return triton_cat