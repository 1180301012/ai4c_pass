import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Match the pattern: scale -> softmax -> matmul -> permute"""
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.matmul(tmp_1, in_1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return (tmp_3,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_attention_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, M, N, K,
    stride_in0_b, stride_in0_m, stride_in0_n,
    stride_in1_b, stride_in1_n, stride_in1_k,
    stride_out_b, stride_out_k, stride_out_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel for: scale -> softmax -> matmul -> permute
    Input shapes:
        in_0: [B, M, N]
        in_1: [B, N, K]
    Output shape: [B, K, M] (after permute)
    """
    # Get batch and output position
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Masks
    mask_m = offs_m < M
    mask_k = offs_k < K
    
    # Pointers for this batch
    in_0_batch_ptr = in_0_ptr + pid_b * stride_in0_b
    in_1_batch_ptr = in_1_ptr + pid_b * stride_in1_b
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    # Process each row
    for m_idx in range(BLOCK_M):
        m = pid_m * BLOCK_M + m_idx
        if m >= M:
            break
        
        # Load and scale row from in_0
        row_ptr = in_0_batch_ptr + m * stride_in0_m + offs_n * stride_in0_n
        
        # Process in chunks for softmax
        max_val = float('-inf')
        
        # First pass: find max for numerical stability
        for n_start in range(0, N, BLOCK_N):
            n_offs = n_start + offs_n
            mask_n = n_offs < N
            
            row_vals = tl.load(row_ptr + n_start * stride_in0_n, mask=mask_n, other=float('-inf'))
            row_vals = row_vals * 0.0625  # Scale
            max_val = tl.maximum(max_val, tl.max(row_vals, axis=0))
        
        # Second pass: compute exp and sum
        exp_sum = 0.0
        for n_start in range(0, N, BLOCK_N):
            n_offs = n_start + offs_n
            mask_n = n_offs < N
            
            row_vals = tl.load(row_ptr + n_start * stride_in0_n, mask=mask_n, other=0.0)
            row_vals = row_vals * 0.0625  # Scale
            exp_vals = tl.exp(row_vals - max_val)
            exp_sum += tl.sum(tl.where(mask_n, exp_vals, 0.0), axis=0)
        
        # Third pass: compute softmax and matmul
        for n_start in range(0, N, BLOCK_N):
            n_offs = n_start + offs_n
            mask_n = n_offs < N
            
            # Load and compute softmax values
            row_vals = tl.load(row_ptr + n_start * stride_in0_n, mask=mask_n, other=0.0)
            row_vals = row_vals * 0.0625  # Scale
            softmax_vals = tl.exp(row_vals - max_val) / exp_sum
            softmax_vals = tl.where(mask_n, softmax_vals, 0.0)
            
            # Load corresponding columns from in_1 and accumulate
            in_1_ptrs = in_1_batch_ptr + (n_start + offs_n[:, None]) * stride_in1_n + offs_k[None, :] * stride_in1_k
            mask_nk = mask_n[:, None] & mask_k[None, :]
            in_1_vals = tl.load(in_1_ptrs, mask=mask_nk, other=0.0)
            
            # Accumulate: softmax_vals [BLOCK_N] @ in_1_vals [BLOCK_N, BLOCK_K]
            acc[m_idx, :] += tl.sum(softmax_vals[:, None] * in_1_vals, axis=0)
    
    # Store result with permutation: output is [B, K, M] instead of [B, M, K]
    out_batch_ptr = out_ptr + pid_b * stride_out_b
    out_ptrs = out_batch_ptr + offs_k[:, None] * stride_out_k + offs_m[None, :] * stride_out_m
    mask_km = mask_k[:, None] & mask_m[None, :]
    
    tl.store(out_ptrs, acc.trans(), mask=mask_km)


@torch.fx.wrap
def fused_scale_softmax_matmul_permute(in_0, in_1):
    """
    Fused implementation of: scale -> softmax -> matmul -> permute
    """
    B, M, N = in_0.shape
    _, _, K = in_1.shape
    
    # Output shape after permute: [B, K, M]
    out = torch.empty((B, K, M), dtype=in_0.dtype, device=in_0.device)
    
    # Define grid
    BLOCK_M = 64
    BLOCK_K = 64
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    
    fused_attention_kernel[grid](
        in_0, in_1, out,
        B, M, N, K,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )
    
    return out


def replacement_func():
    return fused_scale_softmax_matmul_permute