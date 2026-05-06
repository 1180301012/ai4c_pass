import torch
import triton
import triton.language as tl

def pattern(query, key, value, attn_mask):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )

def replacement_args(query, key, value, attn_mask):
    return (query, key, value, attn_mask)

@triton.jit
def sdpa_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    attn_mask_ptr,
    out_ptr,
    query_shape,
    key_shape,
    value_shape,
    attn_mask_shape,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate valid indices for this block
    valid_mask = offsets < query_shape[2]
    
    # Load query and key
    q = tl.load(query_ptr + block_start, mask=valid_mask, other=0.0)
    k = tl.load(key_ptr + block_start, mask=valid_mask, other=0.0)
    
    # Compute attention scores
    att_scores = tl.dot(q, k)  
    
    # Apply attention mask (masking out invalid positions)
    mask_val = tl.load(attn_mask_ptr + block_start, mask=valid_mask)
    att_scores = tl.where(mask_val > 0, att_scores, -1e9)  
    
    # Scale scores
    att_scores = tl.exp(att_scores) / tl.sum(tl.exp(att_scores), axis=-1, keepdim=True)  
    
    # Compute output
    out = tl.dot(att_scores, tl.load(value_ptr + block_start, mask=valid_mask))
    
    # Store result
    tl.store(out_ptr + block_start, out, mask=valid_mask)

@torch.fx.wrap
def sdpa_kernel_wrapper(query, key, value, attn_mask):
    batch_size, num_heads, seq_len, head_dim = query.shape
    out_shape = (batch_size, num_heads, seq_len, head_dim)
    out = torch.empty(out_shape, device=query.device, dtype=query.dtype)
    
    sdpa_kernel[(batch_size * num_heads,)](
        query_ptr=query,
        key_ptr=key,
        value_ptr=value,
        attn_mask_ptr=attn_mask,
        out_ptr=out,
        query_shape=torch.tensor([batch_size, num_heads, seq_len, head_dim]),
        key_shape=torch.tensor([batch_size, num_heads, seq_len, head_dim]),
        value_shape=torch.tensor([batch_size, num_heads, seq_len, head_dim]),
        attn_mask_shape=torch.tensor([1, 1, seq_len, seq_len]),
        BLOCK_SIZE=256,
    )
    return out

def replacement_func():
    return sdpa_kernel_wrapper