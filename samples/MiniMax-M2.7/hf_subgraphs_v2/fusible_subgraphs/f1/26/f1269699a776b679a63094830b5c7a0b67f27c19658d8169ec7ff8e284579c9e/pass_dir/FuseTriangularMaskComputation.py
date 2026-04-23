import torch
import triton
import triton.language as tl


@triton.jit
def triangular_mask_kernel(
    seq_len: tl.constexpr,
    tmp_11_ptr,
    tmp_13_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuse the triangular mask computation into a single kernel.
    
    Original computation:
    tmp_10 = torch.arange(seq_len, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]  # [N, 1]
    tmp_12 = torch.arange(seq_len, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]  # [1, N]
    tmp_14 = tmp_13 - tmp_11  # Shape [N, N]
    tmp_15 = -tmp_14  # Negate (becomes causal mask pattern)
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781  # / ln(10)
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_19 += tmp_31
    """
    # Calculate grid
    pid = tl.program_id(0)
    num_pid_in_group = tl.num_programs(0) // seq_len
    group_id = pid // num_pid_in_group
    start_pid = group_id * num_pid_in_group
    
    # Load row and column indices for this position
    row_idx = (pid - start_pid) // seq_len
    col_offset = pid % (seq_len * seq_len)
    row = col_offset // seq_len
    col = col_offset % seq_len
    
    # Calculate tmp_14 = col - row (broadcasting equivalent)
    # tmp_11 has shape [seq_len, 1], tmp_13 has shape [1, seq_len]
    # tmp_14[i, j] = tmp_13[0, j] - tmp_11[i, 0] = j - i
    diff = col - row  # tmp_14
    
    # tmp_15 = -tmp_14
    neg_diff = -diff  # tmp_15
    
    # tmp_16 = tmp_15 < 0
    is_neg = neg_diff < 0
    
    # tmp_17 = tmp_16.to(torch.int64), tmp_18 = tmp_17 * 16
    offset = tl.where(is_neg, 16, 0)  # tmp_19 base (tmp_17 * 16 + 0)
    
    # tmp_20 = abs(tmp_15) = abs(neg_diff)
    abs_val = tl.abs(neg_diff)  # tmp_20
    
    # tmp_21 = tmp_20 < 8
    in_soft_region = abs_val < 8
    
    # Soft computation: log(abs_val / 8) / ln(10) * 8, converted to int, + 8
    # tmp_22 = abs_val.float()
    # tmp_23 = tmp_22 / 8
    # tmp_24 = log(tmp_23)
    # tmp_25 = tmp_24 / ln(10)
    # tmp_26 = tmp_25 * 8
    # tmp_27 = tmp_26.to(int64)
    # tmp_28 = 8 + tmp_27
    
    # Vectorized soft computation: compute (8 + log(x/8) / ln(10) * 8) for x in [0, 8)
    # Use approximation: for small x, log(x) ~ log(8) + (x-8)/8 - (x-8)^2/(2*64) + ...
    # Actually let's compute it more directly with log
    soft_val = tl.where(in_soft_region,
                        (tl.log(abs_val / 8.0) / 2.772588722239781 + 1.0) * 8.0,
                        16.0)  # For out-of-soft-region, this gets clamped anyway
    
    # Convert to int64: tmp_27 = tmp_26.to(torch.int64)
    soft_val_int = soft_val.to(tl.int64)
    
    # tmp_28 = 8 + tmp_27
    min_val = 8 + soft_val_int  # This is tmp_28
    
    # tmp_30 = min(tmp_28, 15)
    clamped_val = tl.minimum(min_val, 15)  # tmp_30
    
    # tmp_31 = where(tmp_21, tmp_20, tmp_30)
    final_soft = tl.where(in_soft_region, abs_val, clamped_val)  # tmp_31
    
    # tmp_32 = tmp_19 + tmp_31 = offset + final_soft
    result = offset + final_soft
    
    # Store result
    off = pid * seq_len * seq_len + row * seq_len + col
    tl.store(out_ptr + off, result)


@torch.fx.wrap
def fused_triangular_mask(seq_len: int) -> torch.Tensor:
    """Fused triangular mask computation kernel wrapper."""
    N = seq_len
    total_elements = N * N
    
    # Allocate output tensor
    out = torch.empty((N, N), dtype=torch.int64, device='cuda')
    
    # Launch kernel with proper grid
    grid = (total_elements,)
    BLOCK_SIZE = 128
    
    triangular_mask_kernel[grid](
        seq_len=N,
        tmp_11_ptr=0,  # Not used, computed inline
        tmp_13_ptr=0,  # Not used, computed inline
        out_ptr=out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the triangular mask computation pattern.
    
    Returns tmp_9 (dropout output) and tmp_32 (triangular mask).
    """
    # Embeddings
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    
    # Triangular mask computation (pattern varies by sequence length)
    tmp_10 = torch.arange(11, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(11, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_19 += tmp_31
    tmp_32 = tmp_19
    
    return tmp_9, tmp_32


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """Extract arguments needed for replacement."""
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    """Return the fused kernel function."""
    def fused_forward(in_0, in_1, in_2, in_3, in_4, in_5):
        # Embeddings - keep original for now as they're already optimized by PyTorch
        tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
        tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
        tmp_7 = tmp_5 + tmp_6
        tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05)
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
        
        # Get sequence length from input shapes
        seq_len = in_0.shape[-1]
        
        # Fused triangular mask computation
        tmp_32 = fused_triangular_mask(seq_len)
        
        return tmp_9, tmp_32
    
    return fused_forward