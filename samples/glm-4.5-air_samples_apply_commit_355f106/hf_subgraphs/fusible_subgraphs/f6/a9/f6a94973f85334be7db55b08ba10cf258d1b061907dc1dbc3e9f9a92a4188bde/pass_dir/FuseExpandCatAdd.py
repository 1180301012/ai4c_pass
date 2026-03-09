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
def fuse_expand_cat_add_kernel(
    cls_ptr, features_ptr, pos_embed_ptr, out_ptr,
    cls_size, seq_len, hidden_dim,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Calculate positions
    m = tl.program_id(0)
    
    # Load cls_token (repeated for sequence length)
    cls_val = tl.load(cls_ptr)
    cls_seq = tl.full((seq_len, 1), cls_val, dtype=tl.float32)
    
    # Load features and pos_embed
    feats_start = m * seq_len * hidden_dim
    pos_start = m * seq_len * hidden_dim
    
    feats_offset = feats_start + tl.arange(0, seq_len * hidden_dim)
    pos_offset = pos_start + tl.arange(0, seq_len * hidden_dim)
    
    feats = tl.load(features_ptr + feats_offset, mask=feats_offset < seq_len * hidden_dim, other=0.0)
    pos = tl.load(pos_embed_ptr + pos_offset, mask=pos_offset < seq_len * hidden_dim, other=0.0)
    
    # Reshape and add
    feats_reshaped = feats.reshape((seq_len * hidden_dim,))
    pos_reshaped = pos.reshape((seq_len * hidden_dim,))
    result = feats_reshaped + pos_reshaped
    
    # Store cls + result
    out_offset = m * ((seq_len + 1) * hidden_dim) + tl.arange(0, seq_len * hidden_dim)
    tl.store(out_ptr + out_offset, result, mask=out_offset < seq_len * hidden_dim)
    
    # Store cls token at beginning
    cls_out_offset = m * ((seq_len + 1) * hidden_dim)
    tl.store(out_ptr + cls_out_offset, cls_val, mask=True)

@torch.fx.wrap
def fuse_expand_cat_add(cls_token, features, pos_embed):
    batch_size = features.shape[0]
    seq_len = features.shape[1]
    hidden_dim = features.shape[2]
    
    total_elements = batch_size * (seq_len + 1) * hidden_dim
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((batch_size, seq_len + 1, hidden_dim), 
                     dtype=features.dtype, device=features.device)
    
    fuse_expand_cat_add_kernel[(num_programs,)](
        cls_ptr=cls_token,
        features_ptr=features,
        pos_embed_ptr=pos_embed,
        out_ptr=out,
        cls_size=cls_token.numel(),
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=32
    )
    
    return out

def replacement_func():
    return fuse_expand_cat_add