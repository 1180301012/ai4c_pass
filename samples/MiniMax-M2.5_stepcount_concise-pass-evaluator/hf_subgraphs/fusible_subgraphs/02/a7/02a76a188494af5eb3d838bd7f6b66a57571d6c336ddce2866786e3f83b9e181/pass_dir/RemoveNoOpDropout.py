import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matching: embedding -> layer_norm -> dropout(p=0.0)
    Dropout with p=0.0 is a no-op that can be eliminated by fusing embedding+layer_norm.
    """
    # Embedding lookup
    tmp_3 = torch.nn.functional.embedding(in_0, in_2, 50283, None, 2.0, False, False)
    # Layer normalization
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, None, 1e-05)
    # Dropout with p=0.0 is a no-op - skip it
    tmp_5 = torch.nn.functional.dropout(tmp_4, 0.0, False, False)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement.
    """
    return (in_0, in_1, in_2)


# Autotune configs - use power of 2 BLOCK_SIZE
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
    ],
    key=['num_tokens'],
)
@triton.jit
def embedding_layernorm_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    layer_norm_weight_ptr,
    output_ptr,
    num_tokens,
    vocab_size,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused embedding lookup + layer norm kernel.
    Uses BLOCK_SIZE (power of 2) for triton compatibility.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens
    
    # Load input_ids
    input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
    
    # Use power of 2 for arange (next power of 2 >= 768 is 1024)
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_offsets < embedding_dim
    
    # Compute sum and sum_sq for mean/variance
    sum_val = tl.zeros((BLOCK_SIZE,), tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE,), tl.float32)
    
    # Load embeddings and compute statistics
    # Using BLOCK_SIZE for inner loop to make it power of 2
    for dim_idx in range(0, embedding_dim, 1):
        # Calculate offset into embedding table
        emb_offset = input_ids * embedding_dim + dim_idx
        emb_val = tl.load(embedding_weight_ptr + emb_offset, mask=mask, other=0.0)
        sum_val += emb_val
        sum_sq += emb_val * emb_val
    
    # Compute mean and variance
    mean = sum_val / embedding_dim
    var = (sum_sq / embedding_dim) - (mean * mean) + 1e-05
    std = tl.sqrt(var)
    
    # Load layer norm weight using power of 2 BLOCK_SIZE
    ln_weight = tl.load(layer_norm_weight_ptr + dim_offsets, mask=dim_mask, other=1.0)
    
    # Compute normalized output
    for dim_idx in range(0, embedding_dim, 1):
        emb_offset = input_ids * embedding_dim + dim_idx
        emb_val = tl.load(embedding_weight_ptr + emb_offset, mask=mask, other=0.0)
        normalized = (emb_val - mean) / std
        scaled = normalized * ln_weight
        
        out_offset = offsets * embedding_dim + dim_idx
        tl.store(output_ptr + out_offset, scaled, mask=mask)


@torch.fx.wrap
def fused_kernel(input_ids, layer_norm_weight, embedding_weight):
    """
    Fused embedding lookup + layer normalization.
    """
    num_tokens = input_ids.numel()
    vocab_size = embedding_weight.shape[0]
    embedding_dim = embedding_weight.shape[1]
    
    # Flatten input_ids
    input_ids_flat = input_ids.view(-1)
    
    # Allocate output
    output = torch.empty((num_tokens, embedding_dim), dtype=torch.float32, device=input_ids.device)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(num_tokens, META['BLOCK_SIZE']),)
    
    embedding_layernorm_kernel[grid](
        input_ids_ptr=input_ids_flat,
        embedding_weight_ptr=embedding_weight,
        layer_norm_weight_ptr=layer_norm_weight,
        output_ptr=output,
        num_tokens=num_tokens,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )
    
    return output.view(input_ids.shape[0], input_ids.shape[1], embedding_dim)


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_kernel