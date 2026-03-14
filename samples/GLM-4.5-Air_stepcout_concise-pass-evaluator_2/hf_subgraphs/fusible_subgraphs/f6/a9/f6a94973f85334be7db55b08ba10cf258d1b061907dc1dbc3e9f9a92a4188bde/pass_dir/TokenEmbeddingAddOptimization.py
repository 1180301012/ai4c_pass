import torch
import triton
import triton.language as tl

def pattern(tokens, pos_embed):
    # This matches the operation: tmp_13 = tmp_12 + tmp_6 (tokens + pos_embed)
    return tokens + pos_embed

def replacement_args(tokens, pos_embed):
    return (tokens, pos_embed)

@triton.jit
def token_add_kernel(
    tokens_ptr, pos_embed_ptr, out_ptr,
    seq_len, hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one position in the sequence
    pid = tl.program_id(0)
    
    if pid >= seq_len:
        return
        
    offset = pid * hidden_size
    mask = tl.arange(0, hidden_size) < hidden_size
    
    # Load tokens and positional embedding
    tokens = tl.load(tokens_ptr + offset, mask=mask).to(tl.float32)
    pos_embed = tl.load(pos_embed_ptr + offset, mask=mask).to(tl.float32)
    
    # Add with some optimization - tokens typically don't need special handling
    result = tokens + pos_embed
    
    # Store result
    tl.store(out_ptr + offset, result, mask=mask)

@torch.fx.wrap
def optimized_token_add(tokens, pos_embed):
    batch_size, seq_len, hidden_size = tokens.shape
    
    # Use optimal block size based on hidden size
    if hidden_size >= 768:
        block_size = 256
    else:
        block_size = 128
    
    # Each program handles one sequence position
    num_programs = seq_len
    
    out = torch.empty_like(tokens)
    
    token_add_kernel[(num_programs,)](
        tokens_ptr=tokens,
        pos_embed_ptr=pos_embed,
        out_ptr=out,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=block_size
    )
    
    return out

def replacement_func():
    return optimized_token_add