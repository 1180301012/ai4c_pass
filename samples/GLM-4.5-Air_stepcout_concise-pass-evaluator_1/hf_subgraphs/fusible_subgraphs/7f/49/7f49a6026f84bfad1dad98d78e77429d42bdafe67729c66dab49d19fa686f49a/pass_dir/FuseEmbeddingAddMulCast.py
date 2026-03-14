import torch
import triton
import triton.language as tl


# Pattern matching function - matches the embedding addition, multiply, and cast pattern
def pattern(tmp_4, tmp_5, tmp_0):
    # Two embeddings to add
    tmp_6 = tmp_4 + tmp_5
    # Unsqueeze and multiply with attention mask
    tmp_7 = tmp_0.unsqueeze(-1)
    tmp_8 = tmp_6 * tmp_7
    # Cast to float32
    tmp_9 = tmp_8.to(torch.float32)
    return tmp_9


# Argument extraction function
def replacement_args(tmp_4, tmp_5, tmp_0):
    return (tmp_4, tmp_5, tmp_0)


# Optimized kernel using 2D grid: (batch, seq) with inner loop over hidden dim
@triton.jit
def fused_kernel_2d(
    word_emb_ptr,
    pos_emb_ptr,
    attn_mask_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused kernel: (word_emb + pos_emb) * attn_mask[:, None] -> float32
    
    Grid: (batch_size, seq_len) - one program per (batch, seq) position
    Each program processes all hidden_dim elements for its position
    """
    # Get program coordinates
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    # Compute base offsets
    # word_emb and pos_emb are [batch, seq, hidden]
    base_offset = pid_b * seq_len * hidden_dim + pid_s * hidden_dim
    
    # Load attention mask value (scalar)
    # attn_mask is [batch, seq]
    mask_val = tl.load(attn_mask_ptr + pid_b * seq_len + pid_s)
    
    # Process hidden_dim elements with inner loop
    for off in range(0, hidden_dim, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < hidden_dim
        
        # Compute actual offsets
        word_offs = base_offset + offs
        pos_offs = base_offset + offs
        out_offs = base_offset + offs
        
        # Load embeddings
        word_emb = tl.load(word_emb_ptr + word_offs, mask=mask, other=0.0)
        pos_emb = tl.load(pos_emb_ptr + pos_offs, mask=mask, other=0.0)
        
        # Compute: (word + pos) * mask
        combined = word_emb + pos_emb
        result = combined * mask_val
        
        # Store
        tl.store(output_ptr + out_offs, result, mask=mask)


@torch.fx.wrap
def triton_fused_embedding(word_emb, pos_emb, attn_mask):
    """Fused kernel: embedding addition + mask multiply + cast to float32"""
    batch_size = word_emb.shape[0]
    seq_len = word_emb.shape[1]
    hidden_dim = word_emb.shape[2]
    
    # Allocate output
    output = torch.empty((batch_size, seq_len, hidden_dim), dtype=torch.float32, device=word_emb.device)
    
    # Use full grid dimensions for better parallelism
    # One program per (batch, seq) position
    fused_kernel_2d[(batch_size, seq_len)](
        word_emb,
        pos_emb,
        attn_mask,
        output,
        batch_size,
        seq_len,
        hidden_dim,
        32,  # BLOCK_N
    )
    
    return output


def replacement_func():
    return triton_fused_embedding