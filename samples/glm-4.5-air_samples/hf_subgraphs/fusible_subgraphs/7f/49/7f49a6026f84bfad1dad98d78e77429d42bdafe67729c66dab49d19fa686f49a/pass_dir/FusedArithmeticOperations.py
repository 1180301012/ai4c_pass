import torch
import triton
import triton.language as tl

def pattern(attention_mask, embedding_sum):
    # Pattern matches the remaining operations after embedding computation
    tmp_7 = attention_mask.unsqueeze(-1)
    tmp_8 = embedding_sum * tmp_7
    tmp_9 = tmp_8.to(torch.float32)
    # Return the final result that needs to be observable
    return tmp_9

def replacement_args(attention_mask, embedding_sum):
    return (attention_mask, embedding_sum)

@triton.jit
def fused_arithmetic_kernel(
    attention_mask_ptr,
    embedding_sum_ptr,
    output_ptr,
    num_sequences,
    seq_len,
    num_features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a chunk of tokens
    token_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token_idx < seq_len * num_sequences
    
    # Reshape attention mask to [num_sequences, seq_len, 1]
    flat_token_idx = token_idx
    attention_mask = tl.load(
        attention_mask_ptr + flat_token_idx,
        mask=mask,
        other=1.0  # Default to 1.0 for padding tokens
    )
    
    # Load embeddings and reshape attention mask for broadcasting
    embeddings = tl.load(
        embedding_sum_ptr + flat_token_idx * num_features + tl.arange(0, num_features),
        mask=tl.broadcast_to(mask[:, None], (mask.shape[0], num_features)),
        other=0.0
    )
    
    # Unsqueeze attention mask and multiply with embeddings, then convert to float32
    result = embeddings * attention_mask
    tl.store(
        output_ptr + flat_token_idx * num_features,
        result,
        mask=tl.broadcast_to(mask[:, None], (mask.shape[0], num_features))
    )

@torch.fx.wrap
def fused_arithmetic_operations(attention_mask, embedding_sum):
    num_sequences = attention_mask.shape[0]
    seq_len = attention_mask.shape[1]
    num_features = embedding_sum.shape[2]
    
    # Flatten inputs for processing
    flat_attention_mask = attention_mask.view(-1)
    flat_embedding_sum = embedding_sum.view(-1, num_features)
    
    # Output will be the same shape as embedding_sum
    output_size = (num_sequences, seq_len, num_features)
    output = torch.empty(output_size, dtype=torch.float32, device=attention_mask.device)
    flat_output = output.view(-1, num_features)
    
    BLOCK_SIZE = 512  # Can be tuned
    grid = ((seq_len * num_sequences + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_arithmetic_kernel[grid](
        flat_attention_mask,
        flat_embedding_sum,
        flat_output,
        num_sequences,
        seq_len,
        num_features,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_arithmetic_operations