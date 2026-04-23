import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(arg0):
    """
    Match the computation pattern: scale * softmax * transpose
    The pattern mirrors the model.py operations exactly.
    Softmax is along dim=-1 (last dimension), then transpose(-2, -1) swaps last two dims.
    """
    tmp_0 = arg0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2

# Argument extraction function
def replacement_args(arg0):
    return (arg0,)

# Autotune configuration
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)

@triton.jit
def fused_scale_softmax_transpose_kernel(
    x_ptr,
    output_ptr,
    B,  # Batch size
    H,  # Head dimension
    M,  # Second to last dimension
    N,  # Last dimension (softmax dim)
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: scale + softmax(dim=-1) + transpose(-2, -1)
    
    Input shape: [B, H, M, N] -> output shape: [B, H, N, M]
    
    Each program handles one [M] row of softmax (for fixed b, h, n).
    But since softmax is along dim=-1 (N), we need to parallelize over B*H*M
    and each thread block handles N elements.
    """
    # Get program id
    pid = tl.program_id(0)
    
    # Calculate which (b, h, m) this program handles
    num_rows = B * H * M
    row_id = pid
    
    if row_id >= num_rows:
        return
    
    b = row_id // (H * M)
    h = (row_id // M) % H
    m = row_id % M
    
    # Create offsets for loading N elements for this softmax operation
    row_offset = b * H * M * N + h * M * N + m * N
    offs_n = tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N
    
    # Load the row: shape [N]
    x_row = tl.load(x_ptr + row_offset + offs_n, mask=mask, other=-float('inf'))
    
    # Apply scale
    x_scaled = x_row * scale
    
    # Stable softmax: subtract max
    x_max = tl.max(x_scaled)
    x_exp = tl.exp(x_scaled - x_max)
    x_sum = tl.sum(x_exp)
    softmax_row = x_exp / x_sum
    
    # Store to output [B, H, N, M] - which is transposed
    # output[b, h, n, m] = softmax_row[n]
    out_row_offset = b * H * N * M + h * N * M
    out_offsets = out_row_offset + offs_n * M + m
    out_mask = offs_n < N
    
    tl.store(output_ptr + out_offsets, softmax_row, mask=out_mask)


@torch.fx.wrap
def fused_scale_softmax_transpose(x):
    """
    Fused kernel wrapper: scale + softmax(dim=-1) + transpose(-2, -1)
    Input: [B, H, M, N]
    Output: [B, H, N, M]
    """
    # Get shape info
    assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
    B, H, M, N = x.shape
    
    # Scale factor
    scale = 0.1767766952966369
    
    # Allocate output tensor with transposed shape
    output = torch.empty((B, H, N, M), dtype=x.dtype, device=x.device)
    
    # Grid: one program per row to softmax (B*H*M total)
    grid = (B * H * M,)
    
    fused_scale_softmax_transpose_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        B=B,
        H=H,
        M=M,
        N=N,
        scale=scale,
    )
    
    return output


def next_power_of_2(n):
    """Calculate the smallest power of 2 >= n"""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def replacement_func():
    return fused_scale_softmax_transpose