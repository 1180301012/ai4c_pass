import torch
import triton
import triton.language as tl

def pattern(cls_token, patches, pos_embed):
    tmp_11 = cls_token.expand(1, -1, -1)
    tmp_12 = torch.cat([tmp_11, patches], dim=1)
    tmp_13 = tmp_12 + pos_embed
    return tmp_13

def replacement_args(cls_token, patches, pos_embed):
    return (cls_token, patches, pos_embed)

@triton.jit
def fuse_expand_cat_add_kernel(
    cls_ptr,
    patches_ptr,
    pos_ptr,
    out_ptr,
    cls_size: tl.constexpr,
    patch_seq_len: tl.constexpr,
    embed_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one position in the output sequence
    pid = tl.program_id(0)
    total_len = cls_size + patch_seq_len
    
    # Calculate coordinates
    seq_idx = pid // embed_dim
    embed_idx = pid % embed_dim
    
    mask = seq_idx < total_len
    if not mask:
        return
        
    # Initialize output
    if seq_idx < cls_size:
        # cls token region - copy cls_token
        cls_val = tl.load(cls_ptr + embed_idx)
        pos_val = tl.load(pos_ptr + seq_idx * embed_dim + embed_idx, mask=mask)
        out = cls_val + pos_val
    else:
        # patches region - copy patches and add pos_embed
        patch_idx = seq_idx - cls_size
        patches_val = tl.load(patches_ptr + patch_idx * embed_dim + embed_idx, mask=mask)
        pos_val = tl.load(pos_ptr + seq_idx * embed_dim + embed_idx, mask=mask)
        out = patches_val + pos_val
    
    # Store result
    tl.store(out_ptr + pid, out, mask=mask)

@torch.fx.wrap
def fused_expand_cat_add(cls_token, patches, pos_embed):
    # Get dimensions
    cls_size = cls_token.shape[1]  # Should be 1
    patch_seq_len = patches.shape[1]  # Sequence length from patches
    embed_dim = cls_token.shape[2]  # Embedding dimension
    
    # Create output tensor with combined shape
    output_seq_len = cls_size + patch_seq_len
    out = torch.empty((1, output_seq_len, embed_dim), dtype=cls_token.dtype, device=cls_token.device)
    
    # Reshape for kernel (flatten batch dimension)
    cls_flat = cls_token.reshape(-1)  # [cls_size * embed_dim]
    patches_flat = patches.reshape(-1)  # [patch_seq_len * embed_dim]
    pos_flat = pos_embed.reshape(-1)  # [output_seq_len * embed_dim]
    out_flat = out.reshape(-1)  # [output_seq_len * embed_dim]
    
    # Set up kernel execution
    total_elements = output_seq_len * embed_dim
    grid = (total_elements,)
    
    # Launch kernel
    fuse_expand_cat_add_kernel[grid](
        cls_ptr=cls_flat,
        patches_ptr=patches_flat,
        pos_ptr=pos_flat,
        out_ptr=out_flat,
        cls_size=cls_size,
        patch_seq_len=patch_seq_len,
        embed_dim=embed_dim,
        BLOCK_SIZE=1,
    )
    
    return out

def replacement_func():
    return fused_expand_cat_add