import torch
import triton
import triton.language as tl

def pattern(cls_token, patches, pos_embed):
    """Pattern: expand + cat + add + dropout(p=0.0)"""
    tmp_11 = cls_token.expand(1, -1, -1)
    tmp_12 = torch.cat([tmp_11, patches], dim=1)
    tmp_13 = tmp_12 + pos_embed
    tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False)
    return tmp_14

def replacement_args(cls_token, patches, pos_embed):
    return (cls_token, patches, pos_embed)

@triton.jit
def expand_cat_add_kernel(
    cls_ptr,
    patches_ptr,
    pos_ptr,
    output_ptr,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused expand + cat + add kernel
    cls_token: [1, 1, hidden_dim]
    patches: [1, seq_len-1, hidden_dim]
    pos_embed: [1, seq_len, hidden_dim]
    output: [1, seq_len, hidden_dim]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    n_elements = seq_len * hidden_dim
    mask = offsets < n_elements
    
    # Compute indices
    seq_idx = offsets // hidden_dim
    hidden_idx = offsets % hidden_dim
    
    # Load position embedding
    pos_data = tl.load(pos_ptr + offsets, mask=mask, other=0.0)
    
    # Load from cls_token or patches based on seq_idx
    # If seq_idx == 0: load from cls_token
    # Else: load from patches at position seq_idx - 1
    is_cls = (seq_idx == 0)
    
    # For cls_token: always at position [0, 0, hidden_idx]
    cls_offset = hidden_idx
    
    # For patches: position [0, seq_idx-1, hidden_idx]
    patch_offset = (seq_idx - 1) * hidden_dim + hidden_idx
    
    # Load data conditionally
    cls_data = tl.load(cls_ptr + cls_offset, mask=mask & is_cls, other=0.0)
    patch_data = tl.load(patches_ptr + patch_offset, mask=mask & (~is_cls), other=0.0)
    
    # Select appropriate data
    data = tl.where(is_cls, cls_data, patch_data)
    
    # Add position embedding
    result = data + pos_data
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_expand_cat_add(cls_token, patches, pos_embed):
    """
    Fused implementation of expand + cat + add + dropout(p=0.0)
    cls_token: [1, 1, hidden_dim]
    patches: [1, seq_len-1, hidden_dim]
    pos_embed: [1, seq_len, hidden_dim]
    output: [1, seq_len, hidden_dim]
    """
    batch_size = 1
    patches_seq_len = patches.shape[1]
    hidden_dim = patches.shape[2]
    seq_len = patches_seq_len + 1
    
    output = torch.empty(batch_size, seq_len, hidden_dim, device=patches.device, dtype=patches.dtype)
    
    n_elements = seq_len * hidden_dim
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    expand_cat_add_kernel[grid](
        cls_token,
        patches,
        pos_embed,
        output,
        seq_len,
        hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_expand_cat_add