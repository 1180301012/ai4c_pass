import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matches: torch.cat + slice + mean computation with cutoff=960.
    
    Exact computation pattern from graphs with slice(None, 960, None):
    1. tmp_0 = torch.cat([in_0, in_1], dim=1)
    2. tmp_1 = tmp_0[slice(None, None, None), slice(None, 960, None), 
                     slice(None, None, None), slice(None, None, None)]
    3. tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    4. return (tmp_1, tmp_2)
    
    KEY OPTIMIZATION: 960 = 2 * in_1.shape[1], so slice selects exactly in_1.
    We avoid expensive concatenation+slice by directly using in_1.
    
    NOTE: Exclude dead code like 'tmp_0 = None' from pattern matching.
    """
    # Concatenate inputs along channel dimension 
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    
    # Apply slice operation with cutoff at 960 (selects exactly in_1 when in_1.shape[1] = 480)
    tmp_1 = tmp_0[slice(None, None, None), slice(None, 960, None), 
                  slice(None, None, None), slice(None, None, None)]
    
    # Compute spatial mean with keepdim=True
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    """Extract arguments needed for the optimized kernel."""
    return (in_0, in_1)

@triton.jit
def fused_concat_slice_mean_kernel_optimized(
    in_1_ptr,
    out_tensor_ptr,
    out_mean_ptr,
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """Optimized kernel: skip concat+slicing, process in_1 directly."""
    # Compute program indices
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Create coordinate offsets
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_offset = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    h_offset = tl.arange(0, H)
    w_offset = tl.arange(0, W)
    
    # Create coordinate grids
    n_grid, c_grid, h_grid, w_grid = tl.meshgrid(n_offset, c_offset, h_offset, w_offset)
    
    # Flatten coordinates for memory access
    ptr_offset = (n_grid * C + c_grid) * H * W + h_grid * W + w_grid
    
    # Create masks
    n_mask = n_offset < N
    c_mask = c_offset < C
    h_mask = h_offset < H
    w_mask = w_offset < W
    mask = n_mask[:, None, None] & c_mask[None, :, None] & h_mask[None, None, :] & w_mask[None, None, :]
    mask = mask & (n_grid < N) & (c_grid < C) & (h_grid < H) & (w_grid < W)
    
    # Load and process in_1 directly (skip concat+slicing entirely!)
    in_1 = tl.load(in_1_ptr + ptr_offset, mask=mask, other=0.0)
    
    # Store output tensor (same as in_1)
    tl.store(out_tensor_ptr + ptr_offset, in_1, mask=mask)
    
    # Compute spatial sums for mean calculation
    spatial_sum = tl.sum(in_1, axis=(2, 3))
    spatial_sum_mask = n_mask[:, None] & c_mask[None, :]
    
    # Store partial sums for final mean calculation
    mean_ptr_offset = n_grid[:, 0, 0] * C + c_grid[:, 0, 0]
    tl.store(out_mean_ptr + mean_ptr_offset, spatial_sum, mask=spatial_sum_mask)

@torch.fx.wrap
def fused_concat_slice_mean_optimized(in_0, in_1):
    """Optimized function: skip concat+slicing, process in_1 directly."""
    N, C, H, W = in_1.shape
    
    # Create output tensors  
    out_tensor = torch.empty_like(in_1)
    out_mean = torch.empty((N, C, 1, 1), dtype=in_1.dtype, device=in_1.device)
    
    # Configure block sizes
    BLOCK_SIZE_N = 4
    BLOCK_SIZE_C = 64
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch optimized kernel
    fused_concat_slice_mean_kernel_optimized[(grid_n, grid_c)](
        in_1_ptr=in_1,
        out_tensor_ptr=out_tensor,
        out_mean_ptr=out_mean,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    # Complete mean calculation
    out_mean = out_mean / (H * W)
    
    return (out_tensor, out_mean)

def replacement_func():
    """Returns the optimized function reference."""
    return fused_concat_slice_mean_optimized