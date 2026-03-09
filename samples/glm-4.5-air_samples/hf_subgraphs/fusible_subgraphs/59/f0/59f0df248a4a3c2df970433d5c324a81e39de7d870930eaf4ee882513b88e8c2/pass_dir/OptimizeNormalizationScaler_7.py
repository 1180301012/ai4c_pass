import torch
import triton
import triton.language as tl

def pattern(attention_mask, equality_mask, masked_embedding):
    # Original normalization and scaling computation
    attention_sum = attention_mask.sum(-1)  # tmp_7
    padding_count = equality_mask.sum(-1)   # tmp_9 (after float conversion)
    normalized_count = padding_count.float() / attention_sum  # tmp_11
    scaled_embedding = masked_embedding * 0.88  # tmp_12
    complement = 1 - normalized_count  # tmp_13
    reshaped_complement = complement[slice(None, None, None), None, None]  # tmp_14
    
    return reshaped_complement, scaled_embedding

def replacement_args(attention_mask, equality_mask, masked_embedding):
    return (attention_mask, equality_mask, masked_embedding)

@triton.jit
def normalization_kernel(
    attention_mask_ptr,
    equality_mask_ptr,
    output_ptr,
    num_sequences,
    attention_sum_ptr,
    padding_count_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    offset = seq_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offset < num_sequences
    
    # Load attention mask and sum along sequence dimension
    att_mask = tl.load(attention_mask_ptr + offset, mask=mask, other=0)
    att_sum = tl.sum(att_mask)
    
    # Load equality mask and count padding tokens
    eq_mask = tl.load(equality_mask_ptr + offset, mask=mask, other=0)
    padding_count = tl.sum(eq_mask.to(tl.int32))
    
    # Compute normalized count and complement
    norm_count = padding_count.float() / att_sum
    complement = 1.0 - norm_count
    
    # Store results
    tl.store(attention_sum_ptr + seq_idx, att_sum)
    tl.store(padding_count_ptr + seq_idx, padding_count)
    tl.store(output_ptr + seq_idx, complement)

@triton.jit
def scaling_kernel(
    embedding_ptr,
    output_ptr,
    num_sequences,
    sequence_length,
    embedding_dim,
    scale_factor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pos_idx = tl.program_id(1)
    emb_idx = tl.program_id(2)
    
    offset = (tl.arange(0, BLOCK_SIZE) + pos_idx * sequence_length * embedding_dim + 
              emb_idx * embedding_dim)
    
    mask = tl.arange(0, BLOCK_SIZE) < embedding_dim
    embedding_vals = tl.load(embedding_ptr + offset, mask=mask, other=0.0)
    scaled_vals = embedding_vals * scale_factor
    tl.store(output_ptr + offset, scaled_vals, mask=mask)

@torch.fx.wrap
def optimized_normalization_and_scaling(attention_mask, equality_mask, masked_embedding):
    num_sequences = attention_mask.shape[0]
    
    # Create temporary storage
    attention_sum = torch.empty(num_sequences, dtype=torch.float32, device=attention_mask.device)
    padding_count = torch.empty(num_sequences, dtype=torch.int32, device=attention_mask.device)
    complement = torch.empty(num_sequences, dtype=torch.float32, device=attention_mask.device)
    
    # Launch normalization kernel
    grid = (num_sequences,)
    BLOCK_SIZE = 64
    
    normalization_kernel[grid](
        attention_mask_ptr=attention_mask,
        equality_mask_ptr=equality_mask,
        output_ptr=complement,
        num_sequences=num_sequences,
        attention_sum_ptr=attention_sum,
        padding_count_ptr=padding_count,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape complement for broadcasting: [num_sequences] -> [num_sequences, 1, 1]
    shaped_complement = complement.view(num_sequences, 1, 1)
    
    # Launch scaling kernel for embeddings
    num_sequences, sequence_length, embedding_dim = masked_embedding.shape
    scale_factor = 0.88
    grid2 = (1, sequence_length, embedding_dim)
    
    scaled_embedding = torch.zeros_like(masked_embedding)
    
    scaling_kernel[grid2](
        embedding_ptr=masked_embedding,
        output_ptr=scaled_embedding,
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        scale_factor=scale_factor,
        BLOCK_SIZE=128,
    )
    
    return shaped_complement, scaled_embedding

def replacement_func():
    return optimized_normalization_and_scaling