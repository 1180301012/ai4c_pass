import torch
import triton
import triton.language as tl

def pattern(word_emb, pos_emb):
    """
    Pattern: Addition of two embedding tensors
    word_emb + pos_emb -> sum
    """
    emb_sum = word_emb + pos_emb
    return word_emb, pos_emb, emb_sum

def replacement_args(word_emb, pos_emb):
    return (word_emb, pos_emb)

@triton.jit
def optimized_add_kernel(
    word_emb_ptr, pos_emb_ptr, output_ptr,
    batch_size, seq_len, embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized addition kernel for embedding tensors
    """
    # Each program handles one element
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < batch_size * seq_len * embed_dim
    
    # Load values
    word_val = tl.load(word_emb_ptr + offset, mask=mask, other=0.0)
    pos_val = tl.load(pos_emb_ptr + offset, mask=mask, other=0.0)
    
    # Add and store
    result = word_val + pos_val
    tl.store(output_ptr + offset, result, mask=mask)

@torch.fx.wrap
def optimized_add(word_emb, pos_emb):
    batch_size, seq_len, embed_dim = word_emb.shape
    
    output = torch.empty_like(word_emb)
    
    BLOCK_SIZE = 1024
    num_programs = (batch_size * seq_len * embed_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_add_kernel[(num_programs,)](
        word_emb_ptr=word_emb,
        pos_emb_ptr=pos_emb,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    def optimized_add_wrapper(word_emb, pos_emb):
        # For small tensors, use regular addition (no triton overhead)
        if word_emb.numel() < 1024:
            result = word_emb + pos_emb
        else:
            # Use optimized triton kernel
            result = optimized_add(word_emb, pos_emb)
        
        # Return all three values as required by pattern
        return word_emb, pos_emb, result
    
    return optimized_add_wrapper