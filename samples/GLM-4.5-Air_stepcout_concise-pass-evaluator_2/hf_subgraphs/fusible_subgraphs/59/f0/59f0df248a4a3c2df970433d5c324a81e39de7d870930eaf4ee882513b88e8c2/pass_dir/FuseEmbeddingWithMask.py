import torch
import triton
import triton.language as tl

def pattern(input_ids, weight_table):
    """
    Pattern: Exact operations from model:
    tmp_3 = torch.nn.functional.embedding(tmp_1, tmp_2, 1, None, 2.0, False, False)
    tmp_4 = tmp_1.__eq__(2)
    tmp_5 = tmp_4.unsqueeze(-1)  
    tmp_6 = tmp_3.masked_fill(tmp_5, 0.0)
    Returns exactly what the model returns: (tmp_6, nothing for mask since it's not used)
    """
    # tmp_3 = torch.nn.functional.embedding(tmp_1, tmp_2, 1, None, 2.0, False, False)
    tmp_3 = torch.nn.functional.embedding(input_ids, weight_table, 1, None, 2.0, False, False)
    
    # tmp_4 = tmp_1.__eq__(2)
    tmp_4 = input_ids.__eq__(2)
    
    # tmp_5 = tmp_4.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    
    # tmp_6 = tmp_3.masked_fill(tmp_5, 0.0)
    tmp_6 = tmp_3.masked_fill(tmp_5, 0.0)
    
    # The model later uses tmp_6 and returns it, so we return that
    return tmp_6


def replacement_args(input_ids, weight_table):
    return (input_ids, weight_table)


@triton.jit
def fused_embedding_kernel(
    input_ids_ptr,
    weight_table_ptr,
    output_embeddings_ptr,
    embedding_dims,
    num_embeddings,
    batch_size,
    seq_len,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # Calculate global memory position
    output_offset = batch_idx * seq_len * embedding_dim + seq_idx * embedding_dim
    input_offset = batch_idx * seq_len + seq_idx
    
    # Check bounds
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Load input ID and embedding weight
    input_id = tl.load(input_ids_ptr + input_offset)
    weight_offset = input_id * embedding_dim
    
    # Load embedding vector
    mask = tl.arange(0, BLOCK_SIZE) < embedding_dim
    weights = tl.load(
        weight_table_ptr + weight_offset,
        mask=mask,
        other=0.0
    )
    
    # Check if this is a padding token (value == 2)
    is_padding = input_id == 2
    
    # Apply masking by zeroing out if padding
    output_embeddings = tl.where(is_padding, 0.0, weights)
    
    # Store result
    tl.store(output_embeddings_ptr + output_offset, output_embeddings, mask=mask)


@torch.fx.wrap
def fused_embedding_lookup(input_ids, weight_table):
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = weight_table.shape
    
    # Allocate output
    masked_embeddings = torch.zeros((batch_size, seq_len, embedding_dim), 
                                   dtype=weight_table.dtype, 
                                   device=weight_table.device)
    
    # Calculate grid size
    total_elements = batch_size * seq_len
    BLOCK_SIZE = 128  # Smaller block size for better memory coalescing
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_embedding_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        weight_table_ptr=weight_table,
        output_embeddings_ptr=masked_embeddings,
        embedding_dims=(num_embeddings, embedding_dim),
        num_embeddings=num_embeddings,
        batch_size=batch_size,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return masked_embeddings


def replacement_func():
    return fused_embedding_lookup