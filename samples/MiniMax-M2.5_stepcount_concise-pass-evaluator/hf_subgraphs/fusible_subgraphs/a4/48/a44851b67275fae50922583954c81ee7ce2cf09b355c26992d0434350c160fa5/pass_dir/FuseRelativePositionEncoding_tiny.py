import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching the position encoding computation in ConViT Tiny.
    The computation creates relative position embeddings using:
    - torch.arange to create indices
    - repeat/repeat_interleave to tile the difference matrix
    - Squaring and adding to create the final encoding
    """
    # Layer norm part (we include this to match the full subgraph)
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (192,), tmp_1, tmp_0, 1e-06)
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


# Optimized Triton kernel for relative position encoding generation (for tiny model)
@triton.jit
def relative_pos_encoding_kernel_tiny(
    output_ptr,
    grid_size: tl.constexpr,
    n_cells: tl.constexpr,
):
    """
    Optimized kernel for generating relative position encodings for ConViT Tiny.
    
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
    row = pid // n_cells
    col = pid % n_cells
    
    # Calculate the original indices (0-13) for each dimension
    block_row = row // grid_size
    block_col = col // grid_size
    inner_row = row % grid_size
    inner_col = col % grid_size
    
    # Compute the difference values
    row_diff = inner_row - inner_col
    col_diff = block_col - block_row
    squared_sum = row_diff * row_diff + col_diff * col_diff
    
    # Compute output offsets
    base_offset = row * n_cells + col
    
    # Store all three channels
    tl.store(output_ptr + base_offset, tl.cast(row_diff, tl.float32))
    tl.store(output_ptr + n_cells * n_cells + base_offset, tl.cast(col_diff, tl.float32))
    tl.store(output_ptr + 2 * n_cells * n_cells + base_offset, tl.cast(squared_sum, tl.float32))


@torch.fx.wrap
def optimized_relative_pos_encoding_tiny(in_0, in_1, in_2, in_3):
    """
    Optimized implementation for ConViT Tiny that fuses the position encoding generation
    into a single kernel, while keeping the layer norm as-is.
    """
    # Layer norm computation with 192 features (tiny model)
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (192,), in_1, in_0, 1e-06)
    
    # Optimized position encoding generation using Triton
    output = torch.zeros(1, 196, 196, 3, dtype=torch.float32, device='cuda')
    
    grid_size = 14
    n_cells = 196
    num_programs = n_cells
    
    relative_pos_encoding_kernel_tiny[(num_programs,)](
        output_ptr=output,
        grid_size=grid_size,
        n_cells=n_cells,
    )
    
    return (output, tmp_2, tmp_3)


def replacement_func():
    """Return the optimized function."""
    return optimized_relative_pos_encoding_tiny