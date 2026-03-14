import torch
import triton
import triton.language as tl
from torch import device

def pattern(in_0, in_1):
    """Match the relative position embedding pattern for seq_len=512"""
    tmp_1 = torch.arange(512, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = in_1 - tmp_2
    tmp_4 = tmp_3 + 2048
    tmp_5 = tmp_4 - 1
    tmp_6 = torch.nn.functional.embedding(tmp_5, in_0, None, None, 2.0, False, False)
    tmp_7 = tmp_6.to(dtype=torch.float32)
    return (tmp_7,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_relpos_embedding_kernel(
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    seq_len,
    embed_dim,
    vocab_size,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: pid_pair for (row, col) pairs, pid_k for embed_dim blocks
    pid_pair = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    row = pid_pair // seq_len
    col = pid_pair % seq_len
    
    # Load position_id for this row
    pos_id = tl.load(in_1_ptr + row)
    
    # Compute embedding index: pos_id - col + 2047
    emb_idx = pos_id - col + 2047
    
    # Clamp to valid range [0, vocab_size)
    emb_idx = tl.where(emb_idx < 0, 0, emb_idx)
    emb_idx = tl.where(emb_idx >= vocab_size, vocab_size - 1, emb_idx)
    
    # Load block of embedding vector
    k_start = pid_k * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < embed_dim
    
    emb_ptr = in_0_ptr + emb_idx * embed_dim + offs_k
    emb_vals = tl.load(emb_ptr, mask=mask_k, other=0.0)
    
    # Scale by 2.0
    emb_vals = emb_vals * 2.0
    
    # Store to output
    out_offset = pid_pair * embed_dim + offs_k
    tl.store(out_ptr + out_offset, emb_vals, mask=mask_k)

@torch.fx.wrap
def fused_relpos_embedding(in_0, in_1):
    seq_len = in_1.shape[0]
    embed_dim = in_0.shape[1]
    vocab_size = in_0.shape[0]
    
    output = torch.empty(seq_len, seq_len, embed_dim, dtype=torch.float32, device=in_0.device)
    
    num_pairs = seq_len * seq_len
    BLOCK_K = 64
    
    grid = (num_pairs, (embed_dim + BLOCK_K - 1) // BLOCK_K)
    
    fused_relpos_embedding_kernel[grid](
        in_1,
        in_0,
        output,
        seq_len,
        embed_dim,
        vocab_size,
        BLOCK_K,
    )
    
    return output

def replacement_func():
    return fused_relpos_embedding