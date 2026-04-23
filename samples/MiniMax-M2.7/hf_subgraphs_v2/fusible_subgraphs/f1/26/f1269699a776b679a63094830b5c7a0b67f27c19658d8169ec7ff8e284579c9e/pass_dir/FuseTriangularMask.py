import torch
import triton
import triton.language as tl


@triton.jit
def triangular_mask_kernel(
    out_ptr,
    n_elements,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuse the triangular mask computation into a single kernel.
    
    The mask computation creates a causal attention bias with soft edges.
    For a sequence of length N:
    - Position (i, j) in the mask represents attention from position i to position j
    - The mask has value 16 for j < i (attention is masked)
    - Near the diagonal (|i-j| < 8), there's a soft transition computed via log function
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert flat index to 2D coordinates
    row = offsets // seq_len
    col = offsets % seq_len
    
    # diff = col - row (broadcasting: [1,N] - [N,1] = [N,N])
    diff = col - row
    
    # neg_diff = -diff
    neg_diff = -diff
    
    # offset = 16 if neg_diff < 0 else 0
    offset = tl.where(neg_diff < 0, 16, 0)
    
    # abs_val = abs(neg_diff)
    abs_val = tl.abs(neg_diff)
    
    # in_soft_region = abs_val < 8
    in_soft_region = abs_val < 8
    
    # Soft region: (log(abs_val/8) / ln(10) + 1) * 8, then + 8
    # This gives us the smooth decay from 16 toward 0 near diagonal
    soft_val = (tl.log(abs_val / 8.0) / 2.772588722239781 + 1.0) * 8.0
    soft_val_int = soft_val.to(tl.int64)
    min_val = 8 + soft_val_int
    
    # clamped_val = min(min_val, 15)
    clamped_val = tl.minimum(min_val, 15)
    
    # final_soft = abs_val if in_soft_region else clamped_val
    final_soft = tl.where(in_soft_region, abs_val, clamped_val)
    
    # result = offset + final_soft
    result = offset + final_soft
    
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_triangular_mask(seq_len: int) -> torch.Tensor:
    """Fused triangular mask computation kernel wrapper."""
    N = seq_len
    n_elements = N * N
    
    # Allocate output tensor
    out = torch.empty((N, N), dtype=torch.int64, device='cuda')
    
    # Launch kernel with proper grid
    BLOCK_SIZE = 128
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    triangular_mask_kernel[(num_programs,)](
        out_ptr=out,
        n_elements=n_elements,
        seq_len=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def get_sequence_length_from_tensor(t):
    """Extract the sequence length from a tensor's shape."""
    # Handle both [B, S] and [S] shaped tensors
    shape = t.shape
    if len(shape) >= 2:
        return shape[-1]
    return shape[-1]


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the triangular mask computation pattern.
    
    Returns tmp_9 (dropout output) and tmp_32 (triangular mask).
    This pattern matches the case with sequence length determined from input.
    """
    # Embeddings
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    
    # Triangular mask computation - generalize to any sequence length
    seq_len = get_sequence_length_from_tensor(in_0)
    tmp_10 = torch.arange(seq_len, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(seq_len, dtype=torch.int64)
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
        seq_len = get_sequence_length_from_tensor(in_0)
        
        # Fused triangular mask computation
        tmp_32 = fused_triangular_mask(seq_len)
        
        return tmp_9, tmp_32
    
    return fused_forward