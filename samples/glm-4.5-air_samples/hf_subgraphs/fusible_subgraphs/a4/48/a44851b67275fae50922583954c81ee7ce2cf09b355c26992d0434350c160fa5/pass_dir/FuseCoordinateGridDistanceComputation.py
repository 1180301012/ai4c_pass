import torch
import triton
import triton.language as tl


# Pattern matching: Match ONLY the coordinate grid and distance computation
# This avoids needing to handle layer_norm in the replacement
def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match ONLY the coordinate grid and distance computation.
    This is a standalone computation that doesn't depend on layer_norm output.
    
    We fuse operations from tmp_4 = torch.zeros(...) through the distance computation
    and tensor assignments to tmp_21 = tmp_4.
    """
    # === Coordinate grid generation and distance computation ===
    # Create tensor to store results
    tmp_4 = torch.zeros(1, 196, 196, 3)
    
    # Generate coordinate arrays
    tmp_5 = torch.arange(14)
    tmp_6 = tmp_5.view(1, -1)
    tmp_5 = None
    
    tmp_7 = torch.arange(14)
    tmp_8 = tmp_7.view(-1, 1)
    tmp_7 = None
    
    # Compute difference: (1, 14) - (14, 1) = (14, 14)
    tmp_9 = tmp_6 - tmp_8
    tmp_6 = tmp_8 = None
    
    # Expand to (196, 196) using repeat
    tmp_10 = tmp_9.repeat(14, 14)
    tmp_11 = tmp_9.repeat_interleave(14, dim=0)
    tmp_9 = None
    
    # Expand further using repeat_interleave
    tmp_12 = tmp_11.repeat_interleave(14, dim=1)
    tmp_11 = None
    
    # Square the differences
    tmp_13 = tmp_10 ** 2
    tmp_14 = tmp_12 ** 2
    
    # Sum of squares = squared distance
    tmp_15 = tmp_13 + tmp_14
    tmp_13 = tmp_14 = None
    
    # Add batch dimension
    tmp_16 = tmp_15.unsqueeze(0)
    tmp_15 = None
    
    # Assign to output tensor
    tmp_4[:, :, :, 2] = tmp_16
    tmp_17 = tmp_4
    tmp_16 = tmp_17 = None
    
    # Assign y-distance (tmp_12)
    tmp_18 = tmp_12.unsqueeze(0)
    tmp_12 = None
    tmp_4[:, :, :, 1] = tmp_18
    tmp_19 = tmp_4
    tmp_18 = tmp_19 = None
    
    # Assign x-distance (tmp_10)
    tmp_20 = tmp_10.unsqueeze(0)
    tmp_10 = None
    tmp_4[:, :, :, 0] = tmp_20
    tmp_21 = tmp_4
    tmp_20 = tmp_21 = None
    
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    """No arguments needed for the replacement kernel."""
    return ()


# Optimized Triton kernel for coordinate grid and distance computation
@triton.jit
def fused_distance_kernel(
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    - x_diff = col % 14 - row % 14
    - y_diff = col // 14 - row // 14
    - dist_sq = x_diff^2 + y_diff^2
    
    All in a single kernel, avoiding intermediate tensor allocations.
    """
    # Get position in the grid
    pid = tl.program_id(0)
    
    # Compute row and column indices (0-195) using Triton ops
    row = pid // 196
    col = pid % 196
    
    # Compute inner and outer indices for the 14x14 grid
    inner_row = row % 14
    inner_col = col % 14
    outer_row = row // 14
    outer_col = col // 14
    
    # Compute differences
    x_diff = inner_col - inner_row  # tmp_10 value
    y_diff = outer_col - outer_row  # tmp_12 value
    
    # Compute squared distance
    dist_sq = x_diff * x_diff + y_diff * y_diff  # tmp_15 value
    
    # Compute output offset: out has shape (1, 196, 196, 3)
    # Using a flat index approach with proper Triton tensor
    flat_idx = row * 196 + col
    
    # Store to 3 consecutive locations starting at flat_idx * 3
    base_ptr = out_ptr + flat_idx * 3
    
    # Store results: [x_diff, y_diff, dist_sq]
    tl.store(base_ptr, x_diff)
    tl.store(base_ptr + 1, y_diff)
    tl.store(base_ptr + 2, dist_sq)


@torch.fx.wrap
def fused_distance_wrapper():
    """
    Wrapper function that launches the fused distance computation kernel.
    This replaces the entire coordinate grid generation and distance computation.
    """
    # Allocate output tensor for the distance grid
    tmp_4 = torch.empty(1, 196, 196, 3, dtype=torch.float32, device='cuda')
    
    # Launch Triton kernel - pass tensor directly, not data_ptr()
    n_elements = 196 * 196  # 38416 elements
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_distance_kernel[(num_programs,)](
        tmp_4,  # Pass tensor directly, not data_ptr()
        n_elements,
        BLOCK_SIZE,
    )
    
    return tmp_4


def replacement_func():
    """Return the replacement function."""
    return fused_distance_wrapper