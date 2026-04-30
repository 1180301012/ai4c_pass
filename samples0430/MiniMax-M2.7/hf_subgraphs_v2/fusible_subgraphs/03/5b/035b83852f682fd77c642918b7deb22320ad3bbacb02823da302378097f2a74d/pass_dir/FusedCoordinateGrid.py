import torch
import triton
import triton.language as tl

# Pattern matching the coordinate grid computation
def pattern():
    # Create coordinate grids using arange(14) -> view -> broadcast subtraction
    tmp_3 = torch.zeros(1, 196, 196, 3)
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_4 = None
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_6 = None
    tmp_8 = tmp_5 - tmp_7
    tmp_5 = tmp_7 = None
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_8 = None
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_10 = None
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    tmp_12 = tmp_13 = None
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_14 = None
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = tmp_15
    setitem = tmp_3
    tmp_15 = setitem = None
    tmp_17 = tmp_11.unsqueeze(0)
    tmp_11 = None
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    setitem_1 = tmp_3
    tmp_17 = setitem_1 = None
    tmp_19 = tmp_9.unsqueeze(0)
    tmp_9 = None
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = tmp_19
    setitem_2 = tmp_3
    tmp_19 = setitem_2 = None
    return tmp_3


def replacement_args():
    return ()


@triton.jit
def coordinate_grid_kernel(output_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Optimized kernel to compute coordinate grids.
    
    Grid: (196, 196, 3) - one program per output element
    output shape: (1, 196, 196, 3)
    
    Math trace:
    - tmp_8[i, j] = j - i for i, j in [0, 13]
    - tmp_9 = tmp_8.repeat(14, 14): tile to 196x196
      tmp_9[row, col] = (col % 14) - (row % 14)
    - tmp_11 = tmp_10.repeat_interleave(14, dim=1) after repeat_interleave(14, dim=0)
      tmp_11[row, col] = (col // 14) - (row // 14)
    
    Channels:
    - ch0: tmp_9 = (col % 14) - (row % 14)
    - ch1: tmp_11 = (col // 14) - (row // 14)
    - ch2: tmp_9^2 + tmp_11^2
    """
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    ch_idx = tl.program_id(2)
    
    # Compute coordinate-based values
    row_mod = row_idx % 14
    col_mod = col_idx % 14
    row_div = row_idx // 14
    col_div = col_idx // 14
    
    # Channel 0: (col % 14) - (row % 14)
    ch0 = col_mod - row_mod
    
    # Channel 1: (col // 14) - (row // 14)
    ch1 = col_div - row_div
    
    # Channel 2: ch0^2 + ch1^2
    ch2 = ch0 * ch0 + ch1 * ch1
    
    # Select correct channel value
    result = tl.where(ch_idx == 0, ch0, tl.where(ch_idx == 1, ch1, ch2))
    
    # Calculate output offset for contiguous memory
    # output has shape (1, 196, 196, 3) with stride (196*196*3, 196*3, 3, 1)
    offset = (row_idx * 196 + col_idx) * 3 + ch_idx
    
    tl.store(output_ptr + offset, result.to(tl.float32))


@torch.fx.wrap
def coordinate_grid_wrapper():
    # Allocate output tensor with same dtype as torch.zeros (float32)
    output = torch.empty((1, 196, 196, 3), dtype=torch.float32, device='cuda')
    
    grid = (196, 196, 3)  # row, col, channel dimensions
    coordinate_grid_kernel[grid](
        output_ptr=output,
        BLOCK_SIZE=1
    )
    
    return output


def replacement_func():
    return coordinate_grid_wrapper