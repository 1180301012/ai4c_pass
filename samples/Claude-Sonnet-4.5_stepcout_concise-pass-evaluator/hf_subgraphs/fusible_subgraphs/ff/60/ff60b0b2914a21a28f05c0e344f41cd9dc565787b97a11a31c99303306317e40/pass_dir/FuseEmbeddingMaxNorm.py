import torch
import triton
import triton.language as tl


def pattern(input_ids, weight):
    """
    Pattern matching for embedding with max_norm=2.0
    """
    # Match the exact embedding operation from model.py
    result = torch.nn.functional.embedding(input_ids, weight, None, None, 2.0, False, False)
    return result


def replacement_args(input_ids, weight):
    return (input_ids, weight)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['embed_dim'],
)
@triton.jit
def fused_embedding_normalize_kernel(
    input_ids_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim,
    vocab_size,
    max_norm,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused embedding lookup with max_norm normalization in a single kernel.
    Each program handles one embedding vector (one token).
    """
    # Get the global token index
    token_idx = tl.program_id(0)
    
    if token_idx >= batch_size * seq_len:
        return
    
    # Load the input id for this token
    input_id = tl.load(input_ids_ptr + token_idx)
    
    # Clamp to valid range
    input_id = tl.maximum(0, tl.minimum(input_id, vocab_size - 1))
    
    # Calculate the base addresses
    weight_base = input_id * embed_dim
    output_base = token_idx * embed_dim
    
    # First pass: load embeddings and compute norm
    norm_sq = 0.0
    num_blocks = tl.cdiv(embed_dim, BLOCK_SIZE)
    
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < embed_dim
        
        # Load embedding vector
        embed_vals = tl.load(weight_ptr + weight_base + offsets, mask=mask, other=0.0)
        
        # Accumulate norm squared
        norm_sq += tl.sum(tl.where(mask, embed_vals * embed_vals, 0.0))
        
        # Store to output
        tl.store(output_ptr + output_base + offsets, embed_vals, mask=mask)
    
    # Compute scale factor
    norm = tl.sqrt(norm_sq + 1e-12)
    scale = tl.minimum(1.0, max_norm / norm)
    
    # Second pass: apply scaling if needed (conditional to save bandwidth)
    if scale < 0.9999:
        for block_idx in range(num_blocks):
            block_start = block_idx * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < embed_dim
            
            # Load and scale
            vals = tl.load(output_ptr + output_base + offsets, mask=mask, other=0.0)
            vals = vals * scale
            tl.store(output_ptr + output_base + offsets, vals, mask=mask)


@torch.fx.wrap
def fused_embedding_maxnorm(input_ids, weight):
    """
    Fused embedding lookup with max_norm constraint.
    """
    # Get dimensions
    batch_size, seq_len = input_ids.shape
    vocab_size, embed_dim = weight.shape
    max_norm = 2.0
    
    # Allocate output
    output = torch.empty((batch_size, seq_len, embed_dim), dtype=weight.dtype, device=weight.device)
    
    # Launch fused kernel
    total_tokens = batch_size * seq_len
    grid = (total_tokens,)
    
    fused_embedding_normalize_kernel[grid](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        max_norm=max_norm,
    )
    
    return output


def replacement_func():
    return fused_embedding_maxnorm