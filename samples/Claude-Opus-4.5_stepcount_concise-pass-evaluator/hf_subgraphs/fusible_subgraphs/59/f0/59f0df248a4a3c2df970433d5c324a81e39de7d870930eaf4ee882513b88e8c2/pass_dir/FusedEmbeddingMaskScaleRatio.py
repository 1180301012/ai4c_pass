import torch
import triton
import triton.language as tl

def pattern(mask, embedded):
    """
    Pattern to match: unsqueeze -> masked_fill -> scale
    mask: boolean mask from eq check [batch, seq_len]
    embedded: embedding result [batch, seq_len, embed_dim]
    """
    mask_expanded = mask.unsqueeze(-1)
    masked_embedded = embedded.masked_fill(mask_expanded, 0.0)
    scaled_embedded = masked_embedded * 0.88
    return scaled_embedded


def replacement_args(mask, embedded):
    return (mask, embedded)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_mask_scale_kernel(
    mask_ptr,
    embedded_ptr,
    output_ptr,
    batch_size,
    seq_len,
    embed_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: mask broadcast + masked_fill + scale by 0.88
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_valid = offsets < n_elements
    
    # Calculate indices
    batch_seq_dim = offsets // embed_dim
    seq_dim_idx = offsets % embed_dim
    batch_idx = batch_seq_dim // seq_len
    seq_idx = batch_seq_dim % seq_len
    
    # Load mask value (same for all embed_dim elements at this batch,seq position)
    mask_offsets = batch_idx * seq_len + seq_idx
    is_masked = tl.load(mask_ptr + mask_offsets, mask=mask_valid, other=False)
    
    # Load embedding values
    embeddings = tl.load(embedded_ptr + offsets, mask=mask_valid, other=0.0)
    
    # Apply mask and scale: if masked -> 0, else -> embedding * 0.88
    scale = 0.88
    result = tl.where(is_masked, 0.0, embeddings * scale)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask_valid)


@torch.fx.wrap
def fused_mask_and_scale(mask, embedded):
    """
    Fused implementation of unsqueeze + masked_fill + scale.
    """
    batch_size, seq_len, embed_dim = embedded.shape
    n_elements = batch_size * seq_len * embed_dim
    
    output = torch.empty_like(embedded)
    
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_mask_scale_kernel[grid](
        mask,
        embedded,
        output,
        batch_size,
        seq_len,
        embed_dim,
        n_elements,
    )
    
    return output


def replacement_func():
    return fused_mask_and_scale