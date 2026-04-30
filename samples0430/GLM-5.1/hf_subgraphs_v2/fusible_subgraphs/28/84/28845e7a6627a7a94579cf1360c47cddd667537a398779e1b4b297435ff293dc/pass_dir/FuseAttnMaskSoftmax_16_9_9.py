import torch
import triton
import triton.language as tl
from torch import device


# ============================================================
# Pattern matching function
# ============================================================
def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device = device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(16, 9, 9)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim = -1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p = 0.1, training = False)
    return tmp_5


# ============================================================
# Argument extraction function
# ============================================================
def replacement_args(in_0, in_1):
    return (in_0, in_1, "16_9_9")


# ============================================================
# Triton kernel - fused attention mask + softmax
# ============================================================
@triton.jit
def fused_attn_softmax_kernel(
    mask_ptr, scores_ptr, out_ptr,
    num_heads, seq_len,
    mask_stride_2, mask_stride_3,
    scores_stride_1, scores_stride_2, scores_stride_3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    1. scores + mask (with broadcasting)
    2. max(result, -FLT_MAX) (clamp)
    3. softmax(dim=-1)
    4. dropout is identity (training=False), so skipped
    
    Each program processes one row of the softmax (one (head, seq_pos) pair).
    """
    row_idx = tl.program_id(0)
    
    head = row_idx // seq_len
    seq_pos = row_idx % seq_len
    
    # Compute row base offsets
    # mask is [1, 1, S, S] - broadcast over heads dimension
    # For row (head, seq_pos), mask row is mask[0, 0, seq_pos, :]
    mask_row_base = seq_pos * mask_stride_2
    
    # scores is [1, H, S, S]
    # For row (head, seq_pos), scores row is scores[0, head, seq_pos, :]
    scores_row_base = head * scores_stride_1 + seq_pos * scores_stride_2
    
    # Output is [H, S, S] contiguous
    # For row_idx = head * S + seq_pos, row base is row_idx * S
    out_row_base = row_idx * seq_len
    
    offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = offsets < seq_len
    
    # Load mask and scores for this row
    mask_vals = tl.load(
        mask_ptr + mask_row_base + offsets * mask_stride_3,
        mask=row_mask,
        other=-3.4028234663852886e+38
    )
    scores_vals = tl.load(
        scores_ptr + scores_row_base + offsets * scores_stride_3,
        mask=row_mask,
        other=0.0
    )
    
    # Step 1: Add mask to scores (broadcasting handled by loading same mask row for all heads)
    x = scores_vals + mask_vals
    
    # Convert to float32 for numerical stability (matches PyTorch type promotion)
    x = x.to(tl.float32)
    
    # Step 2: Clamp (max with -FLT_MAX)
    # In float32, -3.4028234663852886e+38 is approximately -FLT_MAX
    # For float16/bfloat16 inputs, this value is -inf, but after promotion to float32,
    # the clamp operates in float32 where -FLT_MAX is a finite value
    x = tl.maximum(x, -3.4028234663852886e+38)
    
    # Step 3: Softmax on dim=-1 (per row)
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    
    # Exp and normalize
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    result = exp_x / sum_exp
    
    # Store result (Triton handles dtype casting based on output pointer type)
    tl.store(out_ptr + out_row_base + offsets, result, mask=row_mask)


# ============================================================
# Implementation functions for each route
# ============================================================
def _fused_attn_softmax(mask, scores):
    """
    Launch the fused Triton kernel for attention mask + softmax.
    
    Args:
        mask: [1, 1, S, S] attention mask (may have -inf for masked positions)
        scores: [1, H, S, S] attention scores
    
    Returns:
        [H, S, S] softmax output (float32 due to type promotion in torch.max)
    """
    H = scores.shape[1]
    S = scores.shape[2]
    
    # Output is float32 because torch.max(float16/bfloat16, float32_scalar) promotes to float32
    # For float32 inputs, the output is naturally float32
    out = torch.empty((H, S, S), dtype=torch.float32, device=scores.device)
    
    num_rows = H * S
    BLOCK_SIZE = triton.next_power_of_2(S)
    
    grid = (num_rows,)
    
    fused_attn_softmax_kernel[grid](
        mask_ptr=mask,
        scores_ptr=scores,
        out_ptr=out,
        num_heads=H,
        seq_len=S,
        mask_stride_2=mask.stride(2),
        mask_stride_3=mask.stride(3),
        scores_stride_1=scores.stride(1),
        scores_stride_2=scores.stride(2),
        scores_stride_3=scores.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def _placeholder_impl(mask, scores):
    """Placeholder for routes not used in this pass context."""
    return _fused_attn_softmax(mask, scores)


# ============================================================
# Shared dispatch wrapper (identical across all pass files)
# ============================================================
@torch.fx.wrap
def fused_attn_softmax_dispatch(mask, scores, route):
    if route == "16_9_9":
        return _fused_attn_softmax(mask, scores)
    elif route == "12_9_9":
        return _placeholder_impl(mask, scores)
    elif route == "16_13_13":
        return _placeholder_impl(mask, scores)
    else:
        raise ValueError(f"Unknown route: {route}")


# ============================================================
# Replacement function (identical across all pass files)
# ============================================================
def replacement_func():
    return fused_attn_softmax_dispatch