import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching the position encoding computation in ConViT.
    The computation creates relative position embeddings using:
    - torch.arange to create indices
    - repeat/repeat_interleave to tile the difference matrix
    - Squaring and adding to create the final encoding
    """
    # Layer norm part (we include this to match the full subgraph)
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (432,), tmp_1, tmp_0, 1e-06)
    tmp_1 = tmp_0 = None
    
    # Position encoding computation
    tmp_4 = torch.zeros(1, 196, 196, 3)
    tmp_5 = torch.arange(14)
    tmp_6 = tmp_5.view(1, -1)
    tmp_5 = None
    tmp_7 = torch.arange(14)
    tmp_8 = tmp_7.view(-1, 1)
    tmp_7 = None
    tmp_9 = tmp_6 - tmp_8
    tmp_6 = tmp_8 = None
    tmp_10 = tmp_9.repeat(14, 14)
    tmp_11 = tmp_9.repeat_interleave(14, dim=0)
    tmp_9 = None
    tmp_12 = tmp_11.repeat_interleave(14, dim=1)
    tmp_11 = None
    tmp_13 = tmp_10 ** 2
    tmp_14 = tmp_12 ** 2
    tmp_15 = tmp_13 + tmp_14
    tmp_13 = tmp_14 = None
    tmp_16 = tmp_15.unsqueeze(0)
    tmp_15 = None
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_16
    tmp_17 = tmp_4
    tmp_16 = tmp_17 = None
    tmp_18 = tmp_12.unsqueeze(0)
    tmp_12 = None
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_18
    tmp_19 = tmp_4
    tmp_18 = tmp_19 = None
    tmp_20 = tmp_10.unsqueeze(0)
    tmp_10 = None
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_20
    tmp_21 = tmp_4
    tmp_20 = tmp_21 = None
    
    return (tmp_4, tmp_2, tmp_3)


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the optimized replacement."""
    return (in_0, in_1, in_2, in_3)


# Optimized Triton kernel for relative position encoding generation
@triton.jit
def relative_pos_encoding_kernel(
    output_ptr,
    grid_size: tl.constexpr,
    n_cells: tl.constexpr,
):
    """
    Optimized kernel for generating relative position encodings.
    
    This kernel directly computes:
    - Channel 0: row_diff (i - j) for each position pair
    - Channel 1: col_diff (i - j) with different arrangement  
    - Channel 2: row_diff^2 + col_diff^2
    
    grid_size = 14 (the original grid is 14x14)
    n_cells = 196 (14*14)
    Output shape: [1, 196, 196, 3]
    """
    # Get position in the output tensor [1, 196, 196]
    pid = tl.program_id(0)
    
    # Calculate row and column indices in the 196x196 matrix
    row = pid // n_cells  # Which row block (0-13)
    col = pid % n_cells   # Which column position in block (0-195)
    
    # Calculate the original indices (0-13) for each dimension
    block_row = row // grid_size  # Block within row (0-13)
    block_col = col // grid_size  # Block within column (0-13)
    inner_row = row % grid_size   # Position within row block (0-13)
    inner_col = col % grid_size   # Position within column block (0-13)
    
    # Compute the difference values
    # tmp_10 arrangement: repeat the 14x14 matrix 14x14 times
    # So tmp_10[row, col] = (inner_row - inner_col)
    row_diff = inner_row - inner_col
    
    # tmp_12 arrangement: repeat_interleave on both dimensions
    # tmp_12[row, col] = (block_row - block_col) * grid_size + (inner_row - inner_col)
    # Actually let's reconsider: tmp_9 is [14,14] with tmp_9[i,j] = 0 - (j-i) or similar
    # After repeat_interleave: tmp_12[row, col] = tmp_9[block_row + inner_row, block_col + inner_col] 
    # = (inner_row - inner_col)  (if tmp_9[i,j] = i-j)
    
    # Let me trace more carefully:
    # tmp_9 = tmp_6 - tmp_8 where tmp_6 is [1,14] and tmp_8 is [14,1]
    # tmp_9[i,j] = tmp_6[0,j] - tmp_8[i,0] = j - i
    # tmp_10 = tmp_9.repeat(14, 14): tile the 14x14 matrix 14x14 times
    # tmp_10[r, c] = tmp_9[r % 14, c % 14] = (c % 14) - (r % 14) = inner_col - inner_row
    col_diff_arranged = inner_col - inner_row
    
    # tmp_11 = tmp_9.repeat_interleave(14, dim=0): stack 14 copies vertically
    # tmp_11[r, c] = tmp_9[r // 14, c] = (c - r//14)
    # tmp_12 = tmp_11.repeat_interleave(14, dim=1): stack 14 copies horizontally  
    # tmp_12[r, c] = tmp_11[r, c // 14] = (c//14 - r//14)
    # = block_col - block_row
    col_diff = block_col - block_row
    
    # For squared sum, we need the difference in both arrangements
    squared_sum = row_diff * row_diff + col_diff * col_diff
    
    # Compute output offsets
    # output is [1, 196, 196, 3], we need to write to [0, row, col, :]
    base_offset = row * n_cells + col
    
    # Store channel 0: row_diff (tmp_10 arranged)
    tl.store(output_ptr + base_offset, tl.cast(row_diff, tl.float32))
    
    # Store channel 1: col_diff (tmp_12)
    tl.store(output_ptr + n_cells * n_cells + base_offset, tl.cast(col_diff, tl.float32))
    
    # Store channel 2: squared sum
    tl.store(output_ptr + 2 * n_cells * n_cells + base_offset, tl.cast(squared_sum, tl.float32))


@torch.fx.wrap
def optimized_relative_pos_encoding(in_0, in_1, in_2, in_3):
    """
    Optimized implementation that fuses the position encoding generation
    into a single kernel, while keeping the layer norm as-is.
    """
    # Layer norm computation (keep as-is, it's already well optimized)
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (432,), in_1, in_0, 1e-06)
    
    # Optimized position encoding generation using Triton
    # Output shape: [1, 196, 196, 3]
    output = torch.zeros(1, 196, 196, 3, dtype=torch.float32, device='cuda')
    
    # Launch kernel with 196 programs (one per row of the 196x196 output)
    grid_size = 14
    n_cells = 196
    num_programs = n_cells  # 196 programs, one for each row
    
    relative_pos_encoding_kernel[(num_programs,)](
        output_ptr=output,
        grid_size=grid_size,
        n_cells=n_cells,
    )
    
    return (output, tmp_2, tmp_3)


def replacement_func():
    """Return the optimized function."""
    return optimized_relative_pos_encoding