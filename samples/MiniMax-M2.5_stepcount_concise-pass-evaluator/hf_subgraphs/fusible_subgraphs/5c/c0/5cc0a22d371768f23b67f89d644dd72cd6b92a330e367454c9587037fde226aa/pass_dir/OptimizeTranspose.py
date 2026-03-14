import torch
import triton
import triton.language as tl


def pattern(in_2):
    """
    Match the transpose operation: in_2.transpose(-2, -1)
    Input shape: [1, 16, 196, 48]
    Output shape: [1, 16, 48, 196]
    """
    tmp_4 = in_2.transpose(-2, -1)
    return tmp_4


def replacement_args(in_2):
    return (in_2,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
    ],
    key=['feat_dim', 'seq_len'],
)
@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    feat_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Optimized transpose kernel using 2D tiling.
    Input: [batch, num_heads, seq_len, feat_dim]
    Output: [batch, num_heads, feat_dim, seq_len]
    """
    # Calculate program position
    pid = tl.program_id(0)
    
    # Number of tiles in each dimension (for feat and seq)
    num_tiles_m = (feat_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_tiles_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_tiles_per_head = num_tiles_m * num_tiles_n
    num_tiles_total = batch_size * num_heads * num_tiles_per_head
    
    # Bounds check
    if pid >= num_tiles_total:
        return
    
    # Decode pid into batch, head, tile indices
    tile_pid = pid % num_tiles_per_head
    head_idx = (pid // num_tiles_per_head) % num_heads
    batch_idx = pid // (num_tiles_per_head * num_heads)
    
    tile_m = tile_pid // num_tiles_n
    tile_n = tile_pid % num_tiles_n
    
    # Calculate starting positions for this tile
    offs_m = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary conditions
    mask_m = offs_m < feat_dim
    mask_n = offs_n < seq_len
    
    # Initialize output tile in registers
    # Output shape after transpose: [batch, num_heads, feat_dim, seq_len]
    # We need to write to [batch, num_heads, offs_m, offs_n]
    
    # For each element in the tile, load from input and store to output
    # Input: [batch, num_heads, seq_len, feat_dim] -> [batch, num_heads, offs_n, offs_m]
    for n in range(BLOCK_SIZE_N):
        for m in range(BLOCK_SIZE_M):
            seq_idx = tile_n * BLOCK_SIZE_N + n
            feat_idx = tile_m * BLOCK_SIZE_M + m
            
            if seq_idx < seq_len and feat_idx < feat_dim:
                # Source: input[batch, head, seq_idx, feat_idx]
                src_offset = (batch_idx * num_heads * seq_len * feat_dim +
                              head_idx * seq_len * feat_dim +
                              seq_idx * feat_dim + feat_idx)
                
                # Dest: output[batch, head, feat_idx, seq_idx]
                dst_offset = (batch_idx * num_heads * feat_dim * seq_len +
                              head_idx * feat_dim * seq_len +
                              feat_idx * seq_len + seq_idx)
                
                value = tl.load(input_ptr + src_offset)
                tl.store(output_ptr + dst_offset, value)


def optimized_transpose(in_2):
    """
    Optimized transpose using Triton kernel with autotuning.
    Input: [batch, num_heads, seq_len, feat_dim]
    Output: [batch, num_heads, feat_dim, seq_len]
    """
    batch_size, num_heads, seq_len, feat_dim = in_2.shape
    
    # Allocate output
    output = torch.empty((batch_size, num_heads, feat_dim, seq_len), 
                         device=in_2.device, dtype=in_2.dtype)
    
    # Calculate grid
    num_tiles_m = (feat_dim + 8 - 1) // 8  # min BLOCK_SIZE_M
    num_tiles_n = (seq_len + 64 - 1) // 64  # min BLOCK_SIZE_N
    num_tiles_per_head = num_tiles_m * num_tiles_n
    num_programs = batch_size * num_heads * num_tiles_per_head
    
    # Define grid function for autotuning
    def grid(META):
        return (batch_size * num_heads * 
                ((feat_dim + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M']) *
                ((seq_len + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N']),)
    
    # Launch kernel
    transpose_kernel[grid](
        in_2,
        output,
        batch_size,
        num_heads,
        seq_len,
        feat_dim,
    )
    
    return output


# Wrap the function for FX
@torch.fx.wrap
def transpose_wrapper(in_2):
    return optimized_transpose(in_2)


def replacement_func():
    return transpose_wrapper