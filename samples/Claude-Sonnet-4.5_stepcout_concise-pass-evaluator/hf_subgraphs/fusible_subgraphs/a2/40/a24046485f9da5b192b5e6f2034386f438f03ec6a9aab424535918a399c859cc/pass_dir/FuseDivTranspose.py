import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 32, 'BLOCK_D': 8}, num_warps=4),
        triton.Config({'BLOCK_S': 64, 'BLOCK_D': 8}, num_warps=4),
        triton.Config({'BLOCK_S': 128, 'BLOCK_D': 8}, num_warps=8),
        triton.Config({'BLOCK_S': 16, 'BLOCK_D': 8}, num_warps=2),
    ],
    key=['S', 'D'],
)
@triton.jit
def div_transpose_kernel(
    in_ptr,
    out_ptr,
    B, H, S, D,
    divisor,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (cdiv(S, BLOCK_S), cdiv(D, BLOCK_D), B * H)
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)
    
    # Compute batch and head indices
    b = pid_bh // H
    h = pid_bh % H
    
    # Compute block offsets
    s_start = pid_s * BLOCK_S
    d_start = pid_d * BLOCK_D
    
    # Create offset ranges
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    
    # Masks
    s_mask = s_offsets < S
    d_mask = d_offsets < D
    
    # Input base: [B, H, S, D]
    in_base = b * (H * S * D) + h * (S * D)
    
    # Create 2D offsets for input [BLOCK_S, BLOCK_D]
    s_offs_2d = s_offsets[:, None]
    d_offs_2d = d_offsets[None, :]
    in_offsets = in_base + s_offs_2d * D + d_offs_2d
    mask_2d = (s_mask[:, None]) & (d_mask[None, :])
    
    # Load input tile [BLOCK_S, BLOCK_D]
    tile = tl.load(in_ptr + in_offsets, mask=mask_2d, other=0.0)
    
    # Apply division
    tile = tile / divisor
    
    # Transpose tile [BLOCK_S, BLOCK_D] -> [BLOCK_D, BLOCK_S]
    tile_t = tl.trans(tile)
    
    # Output base: [B, H, D, S]
    out_base = b * (H * D * S) + h * (D * S)
    
    # Create 2D offsets for output [BLOCK_D, BLOCK_S]
    d_offs_out = d_offsets[:, None]
    s_offs_out = s_offsets[None, :]
    out_offsets = out_base + d_offs_out * S + s_offs_out
    mask_out = (d_mask[:, None]) & (s_mask[None, :])
    
    # Store output tile [BLOCK_D, BLOCK_S]
    tl.store(out_ptr + out_offsets, tile_t, mask=mask_out)

@torch.fx.wrap
def div_transpose_wrapper(in_0):
    B, H, S, D = in_0.shape
    
    # Ensure input is contiguous
    in_0 = in_0.contiguous()
    
    # Output shape: [B, H, D, S]
    out = torch.empty((B, H, D, S), dtype=in_0.dtype, device=in_0.device)
    
    divisor = 1.6817928305074292
    
    # Grid: (cdiv(S, BLOCK_S), cdiv(D, BLOCK_D), B * H)
    grid = lambda meta: (
        (S + meta['BLOCK_S'] - 1) // meta['BLOCK_S'],
        (D + meta['BLOCK_D'] - 1) // meta['BLOCK_D'],
        B * H,
    )
    
    div_transpose_kernel[grid](
        in_0,
        out,
        B, H, S, D,
        divisor,
    )
    
    return out

def replacement_func():
    return div_transpose_wrapper