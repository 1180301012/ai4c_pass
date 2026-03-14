import torch
import triton
import triton.language as tl


# Fully fuse embedding lookup + add + mask in one kernel
# This eliminates multiple kernel launches and memory transfers

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_embed_mask_kernel(
    input_ids_ptr, position_ids_ptr,
    word_emb_ptr, pos_emb_ptr,
    mask_ptr, output_ptr,
    N, vocab_size, max_pos,
    stride_ids, stride_pos,
    stride_word, stride_pos_emb, stride_mask, stride_out,
    BLOCK_SIZE: tl.constexpr
):
    """Fused embedding lookup + add + mask -> float32"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < N
    
    # Load input_ids and position_ids
    input_ids = tl.load(input_ids_ptr + indices * stride_ids, mask=mask, other=0)
    position_ids = tl.load(position_ids_ptr + indices * stride_pos, mask=mask, other=0)
    
    # Clamp to valid range
    input_ids = tl.minimum(tl.maximum(input_ids, 0), vocab_size - 1)
    position_ids = tl.minimum(tl.maximum(position_ids, 0), max_pos - 1)
    
    # Load word embedding: word_emb[input_ids]
    word_ptrs = word_emb_ptr + input_ids * stride_word
    word_emb = tl.load(word_ptrs, mask=mask, other=0.0)
    
    # Load position embedding: pos_emb[position_ids]
    pos_ptrs = pos_emb_ptr + position_ids * stride_pos_emb
    pos_emb = tl.load(pos_ptrs, mask=mask, other=0.0)
    
    # Add embeddings
    combined = word_emb + pos_emb
    
    # Load mask
    mask_val = tl.load(mask_ptr + indices * stride_mask, mask=mask, other=0.0)
    
    # Multiply by mask and store
    result = combined * mask_val
    tl.store(output_ptr + indices * stride_out, result, mask=mask)


@torch.fx.wrap
def fused_embed_mask_wrapper(input_ids, position_ids, word_emb_weight, pos_emb_weight, mask):
    """Fused: embedding(input_ids) + embedding(position_ids) * mask -> float32"""
    batch = input_ids.size(0)
    seq_len = input_ids.size(1)
    hidden_dim = word_emb_weight.size(1)
    vocab_size = word_emb_weight.size(0)
    max_pos = pos_emb_weight.size(0)
    
    N = batch * seq_len
    
    # Flatten
    input_ids_flat = input_ids.view(N)
    position_ids_flat = position_ids.view(N)
    mask_flat = mask.view(N)
    
    # Output
    output = torch.empty((batch, seq_len, hidden_dim), dtype=torch.float32, device=input_ids.device)
    output_flat = output.view(N, hidden_dim)
    
    # Grid - use more blocks for better occupancy
    grid = ((N + 31) // 32,)
    
    fused_embed_mask_kernel[grid](
        input_ids_ptr=input_ids_flat, position_ids_ptr=position_ids_flat,
        word_emb_ptr=word_emb_weight, pos_emb_ptr=pos_emb_weight,
        mask_ptr=mask_flat, output_ptr=output_flat,
        N=N, vocab_size=vocab_size, max_pos=max_pos,
        stride_ids=1, stride_pos=1,
        stride_word=word_emb_weight.stride(0), 
        stride_pos_emb=pos_emb_weight.stride(0),
        stride_mask=1, stride_out=output_flat.stride(0)
    )
    
    return output


# Pattern - match full computation including embedding lookups
def pattern(input_ids, position_ids, word_emb_weight, pos_emb_weight, mask):
    """Match the full pattern from the model"""
    word_emb = torch.nn.functional.embedding(input_ids, word_emb_weight, 1, None, 2.0, False, False)
    pos_emb = torch.nn.functional.embedding(position_ids, pos_emb_weight, 1, None, 2.0, False, False)
    combined = word_emb + pos_emb
    mask_unsqueezed = mask.unsqueeze(-1)
    masked = combined * mask_unsqueezed
    result = masked.to(torch.float32)
    return result


def replacement_args(input_ids, position_ids, word_emb_weight, pos_emb_weight, mask):
    return (input_ids, position_ids, word_emb_weight, pos_emb_weight, mask)


def replacement_func():
    return fused_embed_mask_wrapper