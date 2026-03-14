import torch
import triton
import triton.language as tl


def pattern(indices, norm_weight, embed_weight):
    emb = torch.nn.functional.embedding(indices, embed_weight, 50283, None, 2.0, False, False)
    normed = torch.nn.functional.layer_norm(emb, (768,), norm_weight, None, 1e-05)
    dropped = torch.nn.functional.dropout(normed, 0.0, False, False)
    return dropped


def replacement_args(indices, norm_weight, embed_weight):
    return (indices, norm_weight, embed_weight)


@triton.jit
def fused_embedding_layernorm_kernel(
    indices_ptr,
    embed_weight_ptr,
    norm_weight_ptr,
    output_ptr,
    embed_stride,
    out_stride,
    HIDDEN_DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    token_idx = tl.load(indices_ptr + row_idx)
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HIDDEN_DIM
    
    # Load embedding and scale
    emb = tl.load(embed_weight_ptr + token_idx * embed_stride + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(norm_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(emb, axis=0) / HIDDEN_DIM
    
    # Center and compute variance
    centered = tl.where(mask, emb - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / HIDDEN_DIM
    
    # Normalize and scale
    rstd = tl.rsqrt(var + EPS)
    output = centered * rstd * scale
    
    # Store result
    tl.store(output_ptr + row_idx * out_stride + offs, output, mask=mask)


@torch.fx.wrap
def fused_embedding_layernorm(indices, norm_weight, embed_weight):
    orig_shape = indices.shape
    n_rows = indices.numel()
    hidden_dim = embed_weight.shape[1]
    
    indices_flat = indices.contiguous().view(-1)
    output = torch.empty(n_rows, hidden_dim, device=indices.device, dtype=embed_weight.dtype)
    
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    fused_embedding_layernorm_kernel[grid](
        indices_flat, embed_weight, norm_weight, output,
        embed_weight.stride(0), output.stride(0),
        HIDDEN_DIM=hidden_dim, EPS=1e-5, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=1,
    )
    
    return output.view(*orig_shape, hidden_dim)


def replacement_func():
    return fused_embedding_layernorm