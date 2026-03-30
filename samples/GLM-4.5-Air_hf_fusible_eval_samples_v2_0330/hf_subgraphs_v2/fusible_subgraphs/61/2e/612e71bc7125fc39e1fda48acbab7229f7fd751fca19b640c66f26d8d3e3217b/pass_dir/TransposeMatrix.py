import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Each program handles a tile
    m = tl.program_id(0)
    k = tl.program_id(1)
    
    # Create offsets within the tile
    offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create mask for bounds checking
    mask_m = offs_m < n_rows
    mask_k = offs_k < n_cols
    
    # Load from source and transpose
    src_ptrs = x_ptr + (offs_m[:, None] * n_cols + offs_k[None, :])
    x = tl.load(src_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Transpose and store
    out_ptrs = out_ptr + (offs_k[:, None] * n_rows + offs_m[None, :])
    tl.store(out_ptrs, x, mask=mask_k[:, None] & mask_m[None, :])

@torch.fx.wrap
def transpose_matrix_triton(x):
    # Input shape: [n_rows, n_cols]
    n_rows, n_cols = x.shape
    
    # Choose tile sizes for better GPU utilization
    BLOCK_SIZE_K = 32  # Tile size in dimension K (columns)
    BLOCK_SIZE_M = 32  # Tile size in dimension M (rows)
    
    # Calculate grid size
    num_blocks_m = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_k = (n_cols + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Output tensor
    out = torch.empty((n_cols, n_rows), dtype=x.dtype, device=x.device)
    
    # Launch transpose kernel
    transpose_kernel[(num_blocks_m, num_blocks_k)](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    return out

def pattern(in_0):
    tmp_2 = in_0.t()
    # Note: The original code has tmp_3 = tmp_2.to(device(type='cuda'))
    # But input is already on cuda, so this is redundant
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return transpose_matrix_triton