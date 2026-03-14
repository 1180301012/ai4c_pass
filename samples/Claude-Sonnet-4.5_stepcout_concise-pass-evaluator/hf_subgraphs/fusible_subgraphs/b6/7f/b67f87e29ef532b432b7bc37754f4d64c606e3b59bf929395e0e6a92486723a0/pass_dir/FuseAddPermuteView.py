import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match the pattern: add + permute(0,2,1) + view
    For graph 1: [1, 9216, 64] -> [1, 64, 96, 96]
    """
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    tmp_2 = tmp_1.view(1, 64, 96, 96)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8),
    ],
    key=['N', 'C'],
)
@triton.jit
def fused_add_permute_kernel_tiled(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized kernel using tiled approach for better memory access patterns
    Input shape: [1, N, C]
    Output shape: [1, C, N] then viewed as [1, C, H, W]
    
    This processes tiles of the data to improve cache locality
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Tile offsets for output space [C, N]
    offs_c = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks
    mask_c = offs_c < C
    mask_n = offs_n < N
    mask = mask_c[:, None] & mask_n[None, :]
    
    # Read from input [1, N, C] with indices [0, offs_n, offs_c]
    # Create 2D tile
    in_ptrs = offs_n[None, :] * C + offs_c[:, None]
    
    # Load data
    in_0_tile = tl.load(in_0_ptr + in_ptrs, mask=mask, other=0.0)
    in_1_tile = tl.load(in_1_ptr + in_ptrs, mask=mask, other=0.0)
    
    # Add
    result = in_0_tile + in_1_tile
    
    # Write to output [1, C, N] with indices [0, offs_c, offs_n]
    out_ptrs = offs_c[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out_ptrs, result, mask=mask)

@torch.fx.wrap
def fused_add_permute_view(in_0, in_1):
    """
    Wrapper for the fused kernel
    """
    B, N, C = in_0.shape
    assert B == 1, "Only batch size 1 is supported"
    
    H = int(N ** 0.5)
    W = H
    
    # Allocate output
    out = torch.empty((B, C, N), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel with 2D grid
    grid = lambda META: (
        triton.cdiv(C, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    fused_add_permute_kernel_tiled[grid](
        in_0, in_1, out,
        N=N, C=C,
    )
    
    # Reshape to final output shape
    out = out.view(B, C, H, W)
    
    return out

def replacement_func():
    return fused_add_permute_view