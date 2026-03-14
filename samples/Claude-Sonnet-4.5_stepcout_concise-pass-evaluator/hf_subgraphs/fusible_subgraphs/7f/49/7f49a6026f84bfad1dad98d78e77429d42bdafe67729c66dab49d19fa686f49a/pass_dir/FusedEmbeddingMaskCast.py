import torch
import triton
import triton.language as tl

def pattern(attention_mask, input_ids, pos_embeddings_weight, word_embeddings_weight, position_ids):
    """
    Pattern to match: two embeddings + add + mask + cast
    """
    # Two embedding lookups with exact same arguments as in model.py
    word_emb = torch.nn.functional.embedding(input_ids, word_embeddings_weight, 1, None, 2.0, False, False)
    pos_emb = torch.nn.functional.embedding(position_ids, pos_embeddings_weight, 1, None, 2.0, False, False)
    
    # Add embeddings
    add_result = word_emb + pos_emb
    
    # Apply attention mask
    mask_unsqueezed = attention_mask.unsqueeze(-1)
    masked = add_result * mask_unsqueezed
    
    # Cast to float32
    output = masked.to(torch.float32)
    
    return output


def replacement_args(attention_mask, input_ids, pos_embeddings_weight, word_embeddings_weight, position_ids):
    return (attention_mask, input_ids, pos_embeddings_weight, word_embeddings_weight, position_ids)


@triton.jit
def fused_embedding_mask_cast_kernel(
    # Input pointers
    attention_mask_ptr,
    input_ids_ptr,
    position_ids_ptr,
    word_emb_weight_ptr,
    pos_emb_weight_ptr,
    output_ptr,
    # Shapes
    total_elements,
    embed_dim,
):
    # Each program processes one position
    pid = tl.program_id(0)
    
    if pid >= total_elements:
        return
    
    # Load scalar indices
    word_id = tl.load(input_ids_ptr + pid)
    pos_id = tl.load(position_ids_ptr + pid)
    mask_val = tl.load(attention_mask_ptr + pid).to(tl.float32)
    
    # Load embeddings as vectors (embed_dim=32, perfect for vectorization)
    embed_offsets = tl.arange(0, 32)
    
    word_emb = tl.load(word_emb_weight_ptr + word_id * 32 + embed_offsets)
    pos_emb = tl.load(pos_emb_weight_ptr + pos_id * 32 + embed_offsets)
    
    # Fused: add + mask + cast (all in one operation)
    result = (word_emb + pos_emb) * mask_val
    
    # Store result
    tl.store(output_ptr + pid * 32 + embed_offsets, result)


@torch.fx.wrap
def fused_embedding_mask_cast(attention_mask, input_ids, pos_embeddings_weight, word_embeddings_weight, position_ids):
    batch_size, seq_len = input_ids.shape
    embed_dim = word_embeddings_weight.shape[1]
    
    # Flatten inputs for contiguous memory access
    input_ids_flat = input_ids.contiguous().view(-1)
    position_ids_flat = position_ids.contiguous().view(-1)
    attention_mask_flat = attention_mask.contiguous().view(-1)
    
    total_elements = batch_size * seq_len
    
    # Create output tensor
    output = torch.empty(total_elements, embed_dim, dtype=torch.float32, device=input_ids.device)
    
    # Launch kernel - one thread per position
    grid = (total_elements,)
    
    fused_embedding_mask_cast_kernel[grid](
        attention_mask_flat,
        input_ids_flat,
        position_ids_flat,
        word_embeddings_weight,
        pos_embeddings_weight,
        output,
        total_elements,
        embed_dim,
    )
    
    return output.view(batch_size, seq_len, embed_dim)


def replacement_func():
    return fused_embedding_mask_cast