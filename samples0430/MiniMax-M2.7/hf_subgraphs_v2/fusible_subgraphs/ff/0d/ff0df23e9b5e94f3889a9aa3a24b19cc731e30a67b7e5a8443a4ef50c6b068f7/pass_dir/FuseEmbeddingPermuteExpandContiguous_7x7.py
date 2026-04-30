import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def fused_embedding_permute_expand_kernel(
    indices_ptr,
    embedding_table_ptr,
    output_ptr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    num_embeddings: tl.constexpr,
    embedding_dim: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    B_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Embedding lookup
    2. Permute [H, W, E] -> [E, H, W]
    3. Unsqueeze + expand [B_out, E, H_in, W_in] -> [B_out, E, H_out, W_out]
    4. Contiguous output
    
    This eliminates intermediate tensor allocations and multiple kernel launches.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B_out * H_out * W_out)
    
    # Decode flat position to (batch, h, w)
    # Total spatial elements per batch
    spatial_elements = H_out * W_out
    
    out_b = offsets // spatial_elements
    out_spatial = offsets % spatial_elements
    out_h = out_spatial // W_out
    out_w = out_spatial % W_out
    
    # Map output position to input position
    # For expand: output position maps to input position via modulo
    in_h = out_h % H_in
    in_w = out_w % W_in
    
    # Calculate index into the flattened input for embedding lookup
    # indices shape is [H_in, W_in], flattened as [H_in * W_in]
    input_idx_flat = in_h * W_in + in_w
    
    # Load the index (int64)
    idx = tl.load(indices_ptr + input_idx_flat).to(tl.int32)
    
    # Clamp index to valid range
    idx = tl.minimum(tl.maximum(idx, 0), num_embeddings - 1)
    
    # Load embedding vector
    # embedding_table has shape [num_embeddings, embedding_dim]
    # We need embedding_table[idx * embedding_dim + e] for each e
    embedding_offset_base = idx * embedding_dim
    
    # Store each element of the embedding vector
    # Output is [B_out, embedding_dim, H_out, W_out]
    # Layout is contiguous: linearize [b, e, h, w] as b*E*H*W + e*H*W + h*W + w
    for e in range(embedding_dim):
        emb_offset = embedding_offset_base + e
        emb_val = tl.load(embedding_table_ptr + emb_offset)
        # Output offset: linearize [b, e, h, w]
        out_offset = (out_b * embedding_dim + e) * spatial_elements + out_h * W_out + out_w
        tl.store(output_ptr + out_offset, emb_val, mask=mask)


@torch.fx.wrap
def fused_embedding_permute_expand(
    embedding_table,  # Shape: [num_embeddings, embedding_dim]
    indices,          # Shape: [H_in, W_in], dtype: int64
    B_out, H_out, W_out,  # Target batch, height, width
    padding_idx=None, # Ignored in optimized path
):
    """
    Fused embedding lookup + permute + unsqueeze + expand + contiguous.
    
    Args:
        embedding_table: [num_embeddings, embedding_dim] tensor
        indices: [H_in, W_in] int64 tensor with indices into embedding table
        B_out, H_out, W_out: Target output batch, height, width
        
    Returns:
        Output tensor of shape [B_out, embedding_dim, H_out, W_out]
    """
    num_embeddings, embedding_dim = embedding_table.shape
    H_in, W_in = indices.shape
    
    # Allocate output: [B_out, embedding_dim, H_out, W_out]
    output = torch.empty(
        (B_out, embedding_dim, H_out, W_out),
        dtype=embedding_table.dtype,
        device=embedding_table.device,
    )
    
    # Launch kernel
    # Total elements = B_out * H_out * W_out
    total_elements = B_out * H_out * W_out
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_embedding_permute_expand_kernel[(num_programs,)](
        indices_ptr=indices,
        embedding_table_ptr=embedding_table,
        output_ptr=output,
        H_out=H_out,
        W_out=W_out,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        H_in=H_in,
        W_in=W_in,
        B_out=B_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1):
    """
    Match the pattern for expand((2, -1, 7, 7)):
    1. in_1.to(device) 
    2. embedding(in_1, in_0)
    3. permute([2, 0, 1])
    4. unsqueeze(0)
    5. expand((2, -1, 7, 7))
    6. contiguous
    """
    tmp_1 = in_1.to(device(type='cuda'))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((2, -1, 7, 7))
    tmp_6 = tmp_5.contiguous()
    
    return tmp_6


def replacement_args(in_0, in_1):
    """
    Extract the arguments needed for the replacement function.
    
    The expand shape (2, -1, 7, 7) maps to B_out=2, H_out=7, W_out=7.
    """
    B_out = 2
    H_out = 7
    W_out = 7
    return (in_0, in_1, B_out, H_out, W_out)


def replacement_func():
    """
    Returns the fused kernel function.
    
    The replacement takes the output of the original pattern and replaces it
    with our optimized fused implementation.
    """
    return fused_embedding_permute_expand