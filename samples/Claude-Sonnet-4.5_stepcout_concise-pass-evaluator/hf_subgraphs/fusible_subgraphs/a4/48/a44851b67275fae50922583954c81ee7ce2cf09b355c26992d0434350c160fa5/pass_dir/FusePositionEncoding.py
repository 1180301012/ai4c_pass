import torch
import triton
import triton.language as tl


def pattern():
    """Pattern: Position encoding computation"""
    tmp_4 = torch.zeros(1, 196, 196, 3)
    tmp_5 = torch.arange(14)
    tmp_6 = tmp_5.view(1, -1)
    tmp_7 = torch.arange(14)
    tmp_8 = tmp_7.view(-1, 1)
    tmp_9 = tmp_6 - tmp_8
    tmp_10 = tmp_9.repeat(14, 14)
    tmp_11 = tmp_9.repeat_interleave(14, dim=0)
    tmp_12 = tmp_11.repeat_interleave(14, dim=1)
    tmp_13 = tmp_10 ** 2
    tmp_14 = tmp_12 ** 2
    tmp_15 = tmp_13 + tmp_14
    tmp_16 = tmp_15.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_16
    tmp_18 = tmp_12.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_18
    tmp_20 = tmp_10.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_20
    return tmp_4


def replacement_args():
    return ()


@triton.jit
def position_encoding_kernel(
    out_ptr,
    GRID_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
):
    """Optimized kernel for position encoding computation"""
    # Each program handles one element in the 196x196 grid
    idx = tl.program_id(0)
    
    if idx >= OUTPUT_SIZE * OUTPUT_SIZE:
        return
    
    # Compute row and column indices in the 196x196 output grid
    row = idx // OUTPUT_SIZE
    col = idx % OUTPUT_SIZE
    
    # Map to the original 14x14 grid
    grid_row = row // GRID_SIZE
    grid_col = col // GRID_SIZE
    
    # Compute position within the tile
    tile_row = row % GRID_SIZE
    tile_col = col % GRID_SIZE
    
    # Compute the relative position offsets
    # tmp_10: horizontal offset (grid_col - grid_row)
    h_offset = grid_col - grid_row
    
    # tmp_12: vertical offset (tile_col - grid_row)
    v_offset = tile_col - grid_row
    
    # tmp_15: squared distance
    sq_dist = h_offset * h_offset + v_offset * v_offset
    
    # Store in the output tensor
    # Shape: [1, 196, 196, 3]
    # Channel 0: h_offset (tmp_10)
    # Channel 1: v_offset (tmp_12)
    # Channel 2: sq_dist (tmp_15)
    
    base_idx = row * OUTPUT_SIZE * 3 + col * 3
    
    tl.store(out_ptr + base_idx + 0, h_offset.to(tl.float32))
    tl.store(out_ptr + base_idx + 1, v_offset.to(tl.float32))
    tl.store(out_ptr + base_idx + 2, sq_dist.to(tl.float32))


@torch.fx.wrap
def optimized_position_encoding():
    """Wrapper for optimized position encoding kernel"""
    GRID_SIZE = 14
    OUTPUT_SIZE = 196  # 14 * 14
    
    # Allocate output tensor on CUDA
    out = torch.zeros(1, OUTPUT_SIZE, OUTPUT_SIZE, 3, dtype=torch.float32, device='cuda')
    
    # Launch kernel
    grid = (OUTPUT_SIZE * OUTPUT_SIZE,)
    position_encoding_kernel[grid](
        out,
        GRID_SIZE=GRID_SIZE,
        OUTPUT_SIZE=OUTPUT_SIZE,
    )
    
    return out


def replacement_func():
    return optimized_position_encoding