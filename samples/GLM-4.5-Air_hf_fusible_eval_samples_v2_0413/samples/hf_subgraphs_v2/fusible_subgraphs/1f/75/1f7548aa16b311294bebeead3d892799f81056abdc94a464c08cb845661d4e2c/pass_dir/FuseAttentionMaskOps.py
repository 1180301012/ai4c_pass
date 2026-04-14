import torch
import triton
import triton.language as tl

# Pattern matching for attention mask processing - lines 10-21
def pattern(attention_mask, cache_pos, range_size):
    # Convert attention mask to boolean
    tmp_2 = attention_mask.to(device=torch.device('cuda:0'), dtype=torch.bool)
    
    # Create range tensors (currently duplicated in original)
    tmp_3 = torch.arange(range_size, device=torch.device('cuda:0'))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_3 = None
    
    # Slice attention mask with range
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_2 = tmp_4 = None
    
    # Create second arange ( Optimization opportunity: this is redundant)
    tmp_6 = torch.arange(range_size, device=torch.device('cuda:0'))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_6 = None
    
    # Reshape cache position and create comparison mask
    tmp_8 = cache_pos.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_7 = tmp_8 = None
    
    # Complex expanding operations
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = None
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_10 = None
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_5 = None
    
    # Final multiplication
    tmp_13 = tmp_11 * tmp_12
    tmp_11 = tmp_12 = None
    
    return tmp_13

# Extract arguments for optimization
def replacement_args(attention_mask, cache_pos, range_size):
    return (attention_mask, cache_pos, range_size)

@triton.jit
def attention_mask_kernel(
    attention_mask_ptr,
    cache_pos_ptr,
    out_ptr,
    batch_size,
    seq_len,
    cache_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a batch element
    batch_idx = tl.program_id(0)
    
    # Create range indices on the fly
    indices = tl.arange(0, seq_len)
    
    # Load attention mask for this batch
    mask_offset = batch_idx * seq_len
    attention_mask = tl.load(attention_mask_ptr + mask_offset + indices, 
                            mask=indices < seq_len, other=False)
    
    # Load cache positions and create causal mask
    cache_pos = tl.load(cache_pos_ptr, mask=indices < cache_len, other=0)
    
    # Create expanded indices for broadcasting
    causal_mask = indices[:, None] <= cache_pos[None, :]
    
    # Expand to match final output shape [1, 1, seq_len, cache_len]
    final_mask = causal_mask[None, None, :, :]
    
    # Store result
    out_idx = batch_idx * seq_len * seq_len
    tl.store(out_ptr + out_idx + indices[:, None] * seq_len + None,
             final_mask, mask=indices[:, None] < seq_len)

@torch.fx.wrap
def optimized_attention_mask(attention_mask, cache_pos, range_size):
    batch_size = attention_mask.shape[0]
    seq_len = attention_mask.shape[1]
    
    # Ensure cache_pos is the right shape
    if cache_pos.dim() == 1:
        cache_pos = cache_pos.view(-1, 1)
    
    # Create output tensor
    out_shape = (batch_size, 1, 1, seq_len, seq_len)
    out = torch.zeros(out_shape, dtype=torch.bool, device='cuda:0')
    
    BLOCK_SIZE = 128
    num_batches = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    attention_mask_kernel[(num_batches,)](
        attention_mask_ptr=attention_mask,
        cache_pos_ptr=cache_pos,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_len=range_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.squeeze(1).squeeze(1)  # Remove singleton dims

# Replacement function
def replacement_func():
    return optimized_attention_mask