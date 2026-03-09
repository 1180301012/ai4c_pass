import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # tmp_0 = in_0
    # tmp_1 = in_1
    # tmp_2 = torch.nn.functional.layer_norm(in_2, (768,), tmp_1, tmp_0, 1e-05)
    tmp_2 = in_2  # Just use input directly for now to isolate transpose issue
    tmp_3 = tmp_2.transpose(-1, -2)
    return (tmp_3,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, seq) position for all features
    program_id = tl.program_id(0)
    batch_idx = program_id // seq_len
    seq_idx = program_id % seq_len
    
    # Get feature offsets
    feat_offsets = tl.arange(0, BLOCK_SIZE)
    feat_mask = feat_offsets < n_features
    
    # Load input data for this (batch, seq) position
    # Input layout: [batch, seq, features] -> batch * seq * features + seq * features + features
    x_offset = batch_idx * seq_len * n_features + seq_idx * n_features + feat_offsets
    x = tl.load(x_ptr + x_offset, mask=feat_mask, other=0.0)
    
    # Store in transposed layout: [batch, features, seq]
    # Output should have shape [batch, features, seq]
    # Map: (batch_idx, seq_idx, feat_offsets) -> (batch_idx, feat_offsets, seq_idx)
    # For contiguous memory layout: batch-major, then feature-major, then seq-major
    out_offset = batch_idx * n_features * seq_len + feat_offsets * seq_len + seq_idx
    tl.store(out_ptr + out_offset, x, mask=feat_mask)

@torch.fx.wrap
def fused_transpose_only(bias, weight, x):
    batch_size = x.shape[0]
    seq_len = x.shape[1] 
    n_features = x.shape[-1]
    
    # Create output tensor with transposed dimensions: [batch, features, seq_len]
    out = torch.empty(batch_size, n_features, seq_len, dtype=x.dtype, device=x.device)
    
    # Optimized block size for features
    BLOCK_SIZE = 128
    if n_features <= 128:
        BLOCK_SIZE = min(128, n_features)
    elif n_features <= 256:
        BLOCK_SIZE = 64
    elif n_features <= 512:
        BLOCK_SIZE = 32
    else:
        BLOCK_SIZE = 16
    
    # Total programs needed = batch_size * seq_len (one program per (batch, seq) position)
    total_programs = batch_size * seq_len
    grid = (triton.cdiv(total_programs, 1),)
    
    fused_transpose_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        n_features=n_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_transpose_only