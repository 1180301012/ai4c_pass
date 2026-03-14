import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['hidden_dim'],
)
@triton.jit
def fused_pre_layernorm_kernel(
    in_5_ptr,
    in_4_ptr,
    embedding_weight_ptr,
    in_6_ptr,
    in_0_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (batch*seq, hidden_blocks)
    pid_batch_seq = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    batch_idx = pid_batch_seq // seq_len
    seq_idx = pid_batch_seq % seq_len
    
    if batch_idx >= batch_size:
        return
    
    # Load scalar values once per thread block
    in_4_val = tl.load(in_4_ptr + batch_idx)
    in_0_val = tl.load(in_0_ptr + batch_idx * seq_len + seq_idx)
    in_6_val = tl.load(in_6_ptr + batch_idx * seq_len + seq_idx)
    
    # Calculate offsets for this block of hidden dimension
    offsets = pid_hidden * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    # Compute base offsets
    base_in_5_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim
    base_embedding_offset = in_6_val * hidden_dim
    
    # Load in_5 and embedding values
    in_5_vals = tl.load(in_5_ptr + base_in_5_offset + offsets, mask=mask, other=0.0)
    embedding_vals = tl.load(embedding_weight_ptr + base_embedding_offset + offsets, mask=mask, other=0.0)
    
    # Fused computation: (in_5 / in_4 + embedding) * in_0
    result = ((in_5_vals / in_4_val) + embedding_vals) * in_0_val
    
    # Store result
    tl.store(out_ptr + base_in_5_offset + offsets, result, mask=mask)


@torch.fx.wrap
def fused_pre_layernorm(in_0, in_1, in_4, in_5, in_6):
    batch_size, seq_len = in_6.shape
    hidden_dim = in_5.shape[2]
    
    out = torch.empty_like(in_5, dtype=torch.float32)
    
    # Use 2D grid for better parallelization
    BLOCK_SIZE = 256
    grid = (batch_size * seq_len, triton.cdiv(hidden_dim, BLOCK_SIZE))
    
    fused_pre_layernorm_kernel[grid](
        in_5,
        in_4,
        in_1,
        in_6,
        in_0,
        out,
        batch_size,
        seq_len,
        hidden_dim,
    )
    
    return out


def pattern(in_0, in_1, in_4, in_5, in_6):
    tmp_4 = in_5 / in_4
    tmp_5 = tmp_4.to(torch.float32)
    tmp_6 = torch.nn.functional.embedding(in_6, in_1, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = in_0.unsqueeze(-1)
    tmp_9 = tmp_7 * tmp_8
    tmp_10 = tmp_9.to(torch.float32)
    return tmp_10


def replacement_args(in_0, in_1, in_4, in_5, in_6):
    return (in_0, in_1, in_4, in_5, in_6)


def replacement_func():
    return fused_pre_layernorm