import torch
import triton
import triton.language as tl

@triton.jit
def fused_view_expand_kernel(
    in_4_ptr, out_ptr,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Grid: (N, K_block) 
    # in_4 has shape [1, 4096, 512] = [1, N, K]
    # out should have shape [1, 4096, 32, 512] = [1, N, H, K] where H = 32
    
    n = tl.program_id(0)
    k_block = tl.program_id(1)
    
    H = 32
    
    # Load in_4[b, n, k]
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # For each (n, k), we need to write to all H positions
    # So out[b, n, h, k] = in_4[b, n, k]
    
    # Load: in_4[0, n, k]
    k_mask = k_offsets < K
    
    # Load values for this n row
    in_4_val = tl.load(in_4_ptr + n * K + k_offsets, mask=k_mask, other=0.0)
    
    # Store to all H positions: out[0, n, h, k]
    # Each thread handles one (n, k) but writes to H locations
    out_base = n * H * K + k_offsets
    
    for h in range(H):
        out_offset = n * H * K + h * K + k_offsets
        tl.store(out_ptr + out_offset, in_4_val, mask=k_mask)


@torch.fx.wrap
def fused_view_expand_wrapper(in_4):
    """
    Fused operation for: unsqueeze(2) + expand([1, 4096, 32, 512])
    in_4: [1, 4096, 512]
    Returns: [1, 4096, 32, 512]
    """
    N, K = 4096, 512
    H = 32
    
    # Allocate output [1, 4096, 32, 512]
    out = torch.empty((1, N, H, K), dtype=in_4.dtype, device=in_4.device)
    
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid = (grid_n, grid_k)
    
    fused_view_expand_kernel[grid](
        in_4, out,
        N, K,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the unsqueeze(2) + expand pattern.
    Returns tmp_8 (expanded tensor) which is observable in the model's return.
    """
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4,)


def replacement_func():
    return fused_view_expand_wrapper