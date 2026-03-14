import torch
import triton
import triton.language as tl


@triton.jit
def fused_layernorm_kernel(
    hidden_ptr, dropout_ptr, weight_ptr,
    out_normalized_ptr, out_weighted_ptr,
    N: tl.constexpr,  # hidden dim (1024)
    M: tl.constexpr,  # seq len * batch 
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Grid: (M,) - each token position gets its own program
    pid = tl.program_id(0)
    
    # Compute row offset
    row_offset = pid * N
    
    # Load weight (broadcast across rows)
    weight_offset = tl.arange(0, BLOCK_SIZE_N)
    weight_mask = weight_mask = weight_offset < N
    weight = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
    
    # Compute sum of squares for this row
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = col_offsets < N
    
    # Load hidden_states + dropout
    hidden_offset = row_offset + col_offsets
    hidden = tl.load(hidden_ptr + hidden_offset, mask=mask, other=0.0)
    dropout = tl.load(dropout_ptr + hidden_offset, mask=mask, other=0.0)
    
    # Add hidden + dropout
    x = hidden + dropout
    
    # Compute x^2 and sum
    x_sq = x * x
    sum_x_sq = tl.sum(x_sq, axis=0)
    
    # Compute rsqrt(mean(x^2) + eps)
    mean_x_sq = sum_x_sq / N
    inv_std = tl.rsqrt(mean_x_sq + eps)
    
    # Normalize: x * inv_std
    normalized = x * inv_std
    
    # Store normalized output (tmp_1)
    tl.store(out_normalized_ptr + hidden_offset, normalized, mask=mask)
    
    # Apply weight: weight * normalized
    weighted = weight * normalized
    
    # Store weighted output (tmp_10)
    tl.store(out_weighted_ptr + hidden_offset, weighted, mask=mask)


def triton_fused_layernorm(hidden, dropout, weight, eps=1e-6):
    """
    Fused layer norm kernel that computes:
    - tmp_1 = hidden + dropout
    - tmp_5 = tmp_1^2
    - tmp_6 = mean(tmp_5, dim=-1, keepdim=True)
    - tmp_7 = tmp_6 + eps
    - tmp_8 = rsqrt(tmp_7)
    - tmp_9 = tmp_1 * tmp_8
    - tmp_10 = weight * tmp_9
    
    Returns (tmp_1, tmp_10)
    """
    B, S, N = hidden.shape
    M = B * S
    
    out_normalized = torch.empty_like(hidden)
    out_weighted = torch.empty_like(hidden)
    
    BLOCK_SIZE_N = 1024
    grid = (M,)
    
    fused_layernorm_kernel[grid](
        hidden, dropout, weight,
        out_normalized, out_weighted,
        N, M, eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out_normalized, out_weighted


def pattern(in_0, in_2, in_3):
    """Match the core computation pattern (skipping dead code from in_1)"""
    tmp_1 = in_3 + in_2
    tmp_4 = tmp_1.to(torch.float32)
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_1 * tmp_8
    tmp_10 = in_0 * tmp_9
    return (tmp_1, tmp_10)


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


@torch.fx.wrap
def triton_fused_layernorm_wrapper(in_0, in_2, in_3):
    """Wrapper that takes 3 args (skipping dead code in_1)"""
    # in_0: weight [N]
    # in_2: dropout [B, S, N]  
    # in_3: hidden_states [B, S, N]
    return triton_fused_layernorm(in_3, in_2, in_0, eps=1e-6)


def replacement_func():
    return triton_fused_layernorm_wrapper