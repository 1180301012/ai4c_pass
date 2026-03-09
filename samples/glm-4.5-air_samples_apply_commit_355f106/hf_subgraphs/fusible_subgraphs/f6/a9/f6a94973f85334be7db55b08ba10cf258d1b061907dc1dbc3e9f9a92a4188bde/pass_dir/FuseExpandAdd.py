import torch
import triton
import triton.language as tl

def pattern(cls_token, features, pos_embed):
    # tmp_11 = tmp_5.expand(1, -1, -1)  
    tmp_11 = cls_token.expand(1, -1, -1)
    # tmp_12 = torch.cat([tmp_11, tmp_10], dim=1)
    tmp_12 = torch.cat([tmp_11, features], dim=1)
    # tmp_13 = tmp_12 + tmp_6
    tmp_13 = tmp_12 + pos_embed
    return tmp_13

def replacement_args(cls_token, features, pos_embed):
    return (cls_token, features, pos_embed)

@triton.jit
def fuse_expand_add_kernel(
    cls_ptr, features_ptr, pos_embed_ptr, out_ptr,
    seq_len, hidden_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one position in the sequence
    pid = tl.program_id(0)
    
    if pid == 0:
        # Handle cls token position
        cls_val = tl.load(cls_ptr)
        start_idx = 0
        end_idx = hidden_dim
        offsets = start_idx + tl.arange(0, hidden_dim)
        
        # Store cls token at position 0
        tl.store(out_ptr + offsets, cls_val, mask=offsets < hidden_dim)
    
    # Handle expanded features + pos_embed
    offsets = pid * hidden_dim + tl.arange(0, hidden_dim)
    mask = offsets < seq_len * hidden_dim
    
    # Load features and position embedding
    features = tl.load(features_ptr + offsets, mask=mask, other=0.0)
    pos = tl.load(pos_embed_ptr + offsets, mask=mask, other=0.0)
    
    # Add them together
    result = features + pos
    
    # Store result (cls token position 0 is already handled)
    storage_offset = (pid + 1) * hidden_dim
    final_offsets = storage_offset + (offsets % hidden_dim)
    tl.store(out_ptr + final_offsets, result, mask=mask)

@torch.fx.wrap
def fuse_expand_add(cls_token, features, pos_embed):
    batch_size = features.shape[0]
    seq_len = features.shape[1]
    hidden_dim = features.shape[2]
    
    # We need to create space for cls token + features
    total_seq_len = seq_len + 1
    
    BLOCK_SIZE = hidden_dim
    num_programs = total_seq_len
    
    out = torch.empty((batch_size, total_seq_len, hidden_dim), 
                     dtype=features.dtype, device=features.device)
    
    fuse_expand_add_kernel[(num_programs,)](
        cls_ptr=cls_token,
        features_ptr=features,
        pos_embed_ptr=pos_embed,
        out_ptr=out,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fuse_expand_add