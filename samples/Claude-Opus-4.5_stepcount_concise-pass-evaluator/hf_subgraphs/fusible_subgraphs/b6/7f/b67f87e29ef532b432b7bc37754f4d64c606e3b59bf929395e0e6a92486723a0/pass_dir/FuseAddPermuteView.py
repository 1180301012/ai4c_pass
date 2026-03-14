import torch
import triton
import triton.language as tl

# Pattern matching function - matches add + permute + view for graph 1
# Input shape: [1, 9216, 64], output: [1, 64, 96, 96]
def pattern(in_0, in_1):
    """
    Match the pattern:
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    tmp_2 = tmp_1.view(1, 64, 96, 96)
    """
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    tmp_2 = tmp_1.view(1, 64, 96, 96)
    return tmp_2


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def add_permute_tiled_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    N: tl.constexpr,  # second dimension (spatial) = 9216
    C: tl.constexpr,  # third dimension (channels) = 64
    TILE_N: tl.constexpr,
    TILE_C: tl.constexpr,
):
    """
    Fused add + permute kernel with 2D tiling.
    Input: [1, N, C]
    Output: [1, C, N]
    
    Uses 2D tiling to improve memory coalescing.
    """
    # Block indices
    pid_n = tl.program_id(0)  # Which tile in N dimension
    pid_c = tl.program_id(1)  # Which tile in C dimension
    
    # Calculate starting positions for this tile
    n_start = pid_n * TILE_N
    c_start = pid_c * TILE_C
    
    # Create offset ranges for the tile
    n_offs = n_start + tl.arange(0, TILE_N)
    c_offs = c_start + tl.arange(0, TILE_C)
    
    # Create 2D grid of indices
    n_idx = n_offs[:, None]  # [TILE_N, 1]
    c_idx = c_offs[None, :]  # [1, TILE_C]
    
    # Masks for boundary conditions
    n_mask = n_idx < N
    c_mask = c_idx < C
    mask = n_mask & c_mask
    
    # Input indices: [n, c] -> linear index = n * C + c
    in_idx = n_idx * C + c_idx
    
    # Load from both inputs (coalesced in C dimension)
    x = tl.load(in0_ptr + in_idx, mask=mask, other=0.0)
    y = tl.load(in1_ptr + in_idx, mask=mask, other=0.0)
    
    # Add
    result = x + y
    
    # Output indices: [c, n] -> linear index = c * N + n
    out_idx = c_idx * N + n_idx
    
    # Store to output (coalesced in N dimension after transpose)
    tl.store(out_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def add_permute_view_fused(in_0, in_1):
    """
    Fused implementation of add + permute + view.
    Output shape: [1, 64, 96, 96]
    """
    # Input shape: [1, 9216, 64]
    batch = in_0.shape[0]
    N = in_0.shape[1]  # spatial dimension (H*W) = 9216
    C = in_0.shape[2]  # channel dimension = 64
    
    # Output shape: [1, 64, 96, 96]
    out = torch.empty((batch, C, 96, 96), dtype=in_0.dtype, device=in_0.device)
    
    # Ensure inputs are contiguous
    in_0_contig = in_0.contiguous()
    in_1_contig = in_1.contiguous()
    
    # Tile sizes - tuned for the specific dimensions
    TILE_N = 64  # Tile along N dimension
    TILE_C = 64  # Tile along C dimension (matches C=64)
    
    # Grid dimensions
    grid_n = triton.cdiv(N, TILE_N)
    grid_c = triton.cdiv(C, TILE_C)
    
    add_permute_tiled_kernel[(grid_n, grid_c)](
        in_0_contig,
        in_1_contig,
        out,
        N,
        C,
        TILE_N,
        TILE_C,
    )
    
    return out


# Replacement function - returns the fused kernel wrapper
def replacement_func():
    return add_permute_view_fused