import torch
import triton
import triton.language as tl

# Pattern matching for the computation: tmp_0.unsqueeze(-1) * result + type conversion
def pattern(tmp_0, tmp_7):
    tmp_8 = tmp_0.unsqueeze(-1)
    tmp_0 = None
    tmp_9 = tmp_7 * tmp_8
    tmp_7 = tmp_8 = None
    tmp_10 = tmp_9.to(torch.float32)
    tmp_9 = None
    return tmp_10

# Argument extraction for the fused pattern
def replacement_args(tmp_0, tmp_7):
    return (tmp_0, tmp_7)

# Triton kernel for fused multiply + unsqueeze + type conversion
@triton.jit
def simple_multiply_kernel(
    attention_mask_ptr,
    embedding_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * seq_len * hidden_dim
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # For multiplication, we need to expand attention_mask from [batch_size, seq_len] to [batch_size, seq_len, hidden_dim]
    # Calculate which element in the flattened tensor this corresponds to
    total_seq = batch_size * seq_len
    token_idx = offsets // hidden_dim  # which token (batch_seq)
    embed_idx = offsets % hidden_dim    # which dimension within the token
    
    # Calculate batch and sequence index from token index
    batch_idx = token_idx // seq_len
    seq_idx = token_idx % seq_len
    
    # Load attention mask (scalar for this token)
    mask_value = tl.load(attention_mask_ptr + batch_idx * seq_len + seq_idx)
    
    # Load embedding value
    embedding_value = tl.load(embedding_ptr + offsets, mask=mask, other=0.0)
    
    # Perform multiplication (broadcasting the mask to hidden dimension)
    result = mask_value * embedding_value
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_multiply_function(attention_mask, summed_embedding):
    batch_size, seq_len = attention_mask.shape
    _, _, hidden_dim = summed_embedding.shape
    
    # Number of elements in the summed embedding tensor
    n_elements = summed_embedding.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(summed_embedding, dtype=torch.float32)
    
    # Launch simplified kernel
    simple_multiply_kernel[(num_programs,)](
        attention_mask_ptr=attention_mask,
        embedding_ptr=summed_embedding,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return simple_multiply_function