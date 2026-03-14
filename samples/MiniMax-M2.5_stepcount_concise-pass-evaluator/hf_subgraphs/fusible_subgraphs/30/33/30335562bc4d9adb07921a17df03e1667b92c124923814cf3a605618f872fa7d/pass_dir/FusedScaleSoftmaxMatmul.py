import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 1}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 2}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 4}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 8}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_kernel(
    query_ptr,      # in_0: [B, N, D]
    value_ptr,      # in_1: [B, D, M]
    output_ptr,     # [B, M, N]
    B, N, D, M,     # dimensions
    scale: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. scale * query
    2. softmax(scale * query, dim=-1)
    3. attention @ value
    4. permute output
    
    All in a single kernel to avoid intermediate tensor creation.
    Output shape: [B, M, N]
    """
    # Get position
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Each program computes output[B, pid_m, :]
    # Which corresponds to the attention-weighted sum over N for a specific M
    
    # Load value for this batch and M position: [D]
    # value has shape [B, D, M], we need value[B, :, pid_m]
    value_offset = pid_b * D * M + pid_m
    value_block_ptr = tl.make_block_ptr(
        value_ptr + pid_b * D * M,
        shape=(D, M),
        strides=(M, 1),
        offsets=(0, pid_m),
        block_shape=(D, 1),
        order=(0, 1),
    )
    value = tl.load(value_block_ptr).to(tl.float32)
    
    # For each block of N, compute softmax-weighted sum
    # We need to compute: output[b, m, n] = sum_d softmax(scale * query)[b, n, d] * value[b, d, m]
    
    # First pass: compute max for numerical stability
    max_val = float('-inf')
    for _ in range(0, N, BLOCK_SIZE_N):
        n_offsets = tl.arange(0, BLOCK_SIZE_N)
        mask = n_offsets < N
        
        # Load query[B, n, :] - need to access query[pid_b, n, :]
        # query shape: [B, N, D]
        query_offset = pid_b * N * D + n_offsets[:, None] * D + tl.arange(0, D)[None, :]
        query = tl.load(query_ptr + query_offset, mask=mask[:, None], other=0.0).to(tl.float32)
        
        # Apply scale
        scaled = query * scale
        
        # Max over D dimension
        row_max = tl.max(scaled, axis=1)
        max_val = tl.maximum(max_val, tl.max(row_max))
    
    # Second pass: compute exp and weighted sum
    output = 0.0
    for _ in range(0, N, BLOCK_SIZE_N):
        n_offsets = tl.arange(0, BLOCK_SIZE_N)
        mask = n_offsets < N
        
        # Load query
        query_offset = pid_b * N * D + n_offsets[:, None] * D + tl.arange(0, D)[None, :]
        query = tl.load(query_ptr + query_offset, mask=mask[:, None], other=0.0).to(tl.float32)
        
        # Apply scale
        scaled = query * scale
        
        # Compute exp(x - max)
        exp_val = tl.exp(scaled - row_max[:, None])
        
        # Sum over D for softmax denominator
        exp_sum = tl.sum(exp_val, axis=1)
        
        # Compute weighted sum: sum_d softmax[d] * value[d]
        # exp_val has shape [BLOCK_SIZE_N, D], value has shape [D]
        weighted = tl.sum(exp_val * value[None, :], axis=1)
        
        # Accumulate output: weighted_sum / exp_sum
        output += weighted / exp_sum
    
    # Store result
    out_offset = pid_b * M * N + pid_m * N + tl.arange(0, N)
    tl.store(output_ptr + out_offset, output, mask=out_offset < B * M * N)


@torch.fx.wrap
def fused_softmax_matmul(query: torch.Tensor, value: torch.Tensor, scale: float = 0.0625) -> torch.Tensor:
    """
    Fused kernel that computes:
    output = (softmax(scale * query, dim=-1) @ value).permute(0, 2, 1)
    
    All in a single kernel for better performance.
    
    Args:
        query: [B, N, D]
        value: [B, D, M]
        scale: scaling factor
    
    Returns:
        output: [B, M, N]
    """
    B, N, D = query.shape
    _, _, M = value.shape
    
    # Allocate output
    output = torch.empty((B, M, N), device=query.device, dtype=query.dtype)
    
    # Grid: (B, M) - each block computes output[b, m, :]
    grid = (B, M)
    
    fused_kernel[grid](
        query, value, output,
        B, N, D, M,
        scale,
    )
    
    return output


def pattern(in_0, in_1):
    """
    Match the pattern: scale * in_0 -> softmax -> matmul -> permute
    
    Args:
        in_0: query tensor [B, N, D]
        in_1: value tensor [B, D, M]
    
    Returns:
        The output after all operations
    """
    scale_factor = 0.0625
    tmp_0 = scale_factor * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.matmul(tmp_1, in_1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    """
    Extract arguments for the replacement function.
    """
    return (in_0, in_1)


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_softmax_matmul