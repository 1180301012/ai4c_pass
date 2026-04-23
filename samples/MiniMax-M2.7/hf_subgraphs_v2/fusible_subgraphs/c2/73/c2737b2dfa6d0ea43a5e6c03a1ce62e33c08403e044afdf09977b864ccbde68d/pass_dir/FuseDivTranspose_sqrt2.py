import torch
import triton
import triton.language as tl

# Pattern matching function - matches div + transpose pattern with sqrt(2) constant
def pattern(in_0):
    """
    Match pattern: in_0 / sqrt(2) constant followed by transpose(-1, -2)
    """
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

# Extract arguments for replacement
def replacement_args(in_0):
    return (in_0, 2.8284271247461903)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def fused_div_transpose_kernel(
    input_ptr,
    output_ptr,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for div + transpose. Simple and correct.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_tile = tl.program_id(2)
    
    # Calculate tile boundaries
    tile_start = pid_tile * BLOCK_SIZE
    tile_end = tl.minimum(tile_start + BLOCK_SIZE, D * N)
    
    # Offsets within tile
    offs = tl.arange(0, BLOCK_SIZE)
    mask = (tile_start + offs) < tile_end
    
    # Position in output [D, N] matrix
    pos = tile_start + offs
    d_idx = pos // N
    n_idx = pos % N
    
    # Input offset: [B, H, N, D] -> n*D + d
    base_input = (pid_b * H + pid_h) * N * D
    input_offset = base_input + n_idx * D + d_idx
    
    # Load, scale, store
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    out = x * scale
    tl.store(output_ptr + base_input + pos, out, mask=mask)


@torch.fx.wrap
def fused_div_transpose_sqrt2(x, scale, route=None):
    """
    Fused division and transpose kernel for sqrt(2) constant.
    """
    B, H, N, D = x.shape
    total = D * N
    
    # Precompute reciprocal
    scale_val = 1.0 / float(scale)
    
    # Allocate output
    output = torch.empty([B, H, D, N], dtype=x.dtype, device=x.device)
    
    # Grid: B x H x num_tiles
    BLOCK_SIZE = 512
    num_tiles = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (B, H, num_tiles)
    
    triton_kernel = fused_div_transpose_kernel
    triton_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        scale=scale_val,
        B=B,
        H=H,
        N=N,
        D=D,
    )
    
    return output


def replacement_func():
    return fused_div_transpose_sqrt2