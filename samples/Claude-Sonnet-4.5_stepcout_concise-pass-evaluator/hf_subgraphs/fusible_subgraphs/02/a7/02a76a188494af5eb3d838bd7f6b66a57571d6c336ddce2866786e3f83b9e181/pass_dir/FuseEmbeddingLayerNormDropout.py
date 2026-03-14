import torch
import triton
import triton.language as tl

def pattern(indices, weight_ln, weight_emb):
    """Match the embedding + layer_norm + dropout pattern"""
    embedded = torch.nn.functional.embedding(indices, weight_emb, 50283, None, 2.0, False, False)
    normalized = torch.nn.functional.layer_norm(embedded, (768,), weight_ln, None, 1e-05)
    dropped = torch.nn.functional.dropout(normalized, 0.0, False, False)
    return dropped

def replacement_args(indices, weight_ln, weight_emb):
    return (indices, weight_ln, weight_emb)

@triton.jit
def fused_embedding_layernorm_kernel(
    indices_ptr,
    weight_emb_ptr,
    weight_ln_ptr,
    output_ptr,
    n_rows,
    hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel that processes entire hidden dimension at once.
    Single memory read per value, compute mean/var/norm in registers.
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Load the embedding index
    idx = tl.load(indices_ptr + row_idx).to(tl.int64)
    
    emb_base = idx * hidden_dim
    output_base = row_idx * hidden_dim
    
    # Load entire embedding vector and layer norm weight in one go
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    x = tl.load(weight_emb_ptr + emb_base + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ln_ptr + offsets, mask=mask, other=1.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / hidden_dim
    
    # Compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_dim
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply weight
    x_norm = x_centered * rstd
    output = x_norm * weight
    
    tl.store(output_ptr + output_base + offsets, output, mask=mask)

@torch.fx.wrap
def fused_embedding_layernorm(indices, weight_ln, weight_emb):
    """
    Wrapper function that launches the fused kernel.
    
    Args:
        indices: Token indices [batch, seq_len]
        weight_ln: Layer norm weight [hidden_dim]
        weight_emb: Embedding weight table [vocab_size, hidden_dim]
    
    Returns:
        Normalized embeddings [batch, seq_len, hidden_dim]
    """
    # Flatten indices for processing
    original_shape = indices.shape
    indices_flat = indices.reshape(-1).contiguous()
    n_rows = indices_flat.shape[0]
    hidden_dim = weight_emb.shape[1]
    
    # Allocate output tensor
    output = torch.empty((n_rows, hidden_dim), device=weight_emb.device, dtype=weight_emb.dtype)
    
    # Use 1024 to load all 768 elements in one go
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    num_warps = 4
    
    fused_embedding_layernorm_kernel[grid](
        indices_flat,
        weight_emb,
        weight_ln,
        output,
        n_rows,
        hidden_dim,
        1e-05,
        BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # Reshape output to match expected shape [batch, seq_len, hidden_dim]
    output_shape = list(original_shape) + [hidden_dim]
    return output.reshape(output_shape)

def replacement_func():
    return fused_embedding_layernorm