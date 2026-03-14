import torch
import triton
import triton.language as tl

def pattern(embedding_sum, attention_mask):
    # The pattern matches addition of embeddings followed by scaling
    # This matches: tmp_6 = tmp_4 + tmp_5 followed by tmp_8 = tmp_6 * tmp_7
    # where tmp_7 = tmp_0.unsqueeze(-1)
    mask_scaled = embedding_sum * attention_mask.unsqueeze(-1)
    return mask_scaled

def replacement_args(embedding_sum, attention_mask):
    return (embedding_sum, attention_mask)

@triton.jit
def add_multiply_kernel(
    embedding_sum_ptr, attention_mask_ptr, output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch x sequence x hidden dimension
    program_id = tl.program_id(0)
    
    # Calculate batch, sequence, and hidden dimension indices
    total_elements = batch_size * seq_len * hidden_dim
    if program_id >= total_elements:
        return
    
    batch_id = program_id // (seq_len * hidden_dim)
    seq_id = (program_id % (seq_len * hidden_dim)) // hidden_dim
    hid_id = program_id % hidden_dim
    
    if batch_id >= batch_size or seq_id >= seq_len:
        return
    
    # Load embedding sum and attention mask
    embedding_sum_val = tl.load(embedding_sum_ptr + program_id)
    attention_val = tl.load(attention_mask_ptr + batch_id * seq_len + seq_id)
    
    # Apply scaling by unsqueezed attention mask (multiply by attention_val for each hidden dimension)
    result = embedding_sum_val * attention_val
    
    # Store result
    tl.store(output_ptr + program_id, result)

@torch.fx.wrap
def fused_add_multiply(embedding_sum, attention_mask):
    # Handle both batched and unbatched cases
    batch_size = 1
    seq_len = embedding_sum.size(-2) if embedding_sum.dim() >= 2 else embedding_sum.size(-1)
    hidden_dim = embedding_sum.size(-1)
    
    if embedding_sum.dim() == 3:
        batch_size, seq_len, hidden_dim = embedding_sum.shape
    elif embedding_sum.dim() == 2:
        seq_len = embedding_sum.size(0)
        hidden_dim = embedding_sum.size(1)
    
    # Create output tensor with same dtype as embedding_sum
    output = torch.zeros_like(embedding_sum)
    
    # Flatten tensors for efficient processing
    if embedding_sum.dim() == 3:
        flat_embedding_sum = embedding_sum.view(-1)
        flat_attention_mask = attention_mask.view(-1)
        flat_output = output.view(-1)
    else:
        flat_embedding_sum = embedding_sum.view(-1)
        flat_attention_mask = attention_mask.view(-1)
        flat_output = output.view(-1)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * seq_len * hidden_dim
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    add_multiply_kernel[(num_programs,)](
        embedding_sum_ptr=flat_embedding_sum,
        attention_mask_ptr=flat_attention_mask,
        output_ptr=flat_output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_add_multiply