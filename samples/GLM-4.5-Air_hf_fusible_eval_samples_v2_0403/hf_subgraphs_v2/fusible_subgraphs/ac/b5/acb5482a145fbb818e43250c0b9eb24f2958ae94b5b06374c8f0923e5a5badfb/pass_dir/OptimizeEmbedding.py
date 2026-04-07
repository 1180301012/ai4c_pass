import torch
import triton
import triton.language as tl

def pattern(tmp_1, tmp_2):
    """Match embedding operation with exact intermediate variable structure"""
    tmp_3 = torch.nn.functional.embedding(tmp_1, tmp_2, None, None, 2.0, False, False)
    return tmp_3

def replacement_args(tmp_1, tmp_2):
    """Extract the input tensors for embedding"""
    return (tmp_1, tmp_2)

@triton.jit
def embedding_lookup_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Custom embedding lookup kernel using Triton"""
    # Get thread IDs
    pid = tl.program_id(0)
    bid = pid // seq_len  # batch index
    sid = pid % seq_len   # sequence index
    
    # Calculate input position
    input_pos = bid * seq_len + sid
    
    # Load input ID with bounds checking
    input_id = tl.load(input_ids_ptr + input_pos)
    
    # Ensure input_id is within valid range
    input_id = tl.where(input_id >= num_embeddings, num_embeddings - 1, input_id)
    
    # Calculate embedding offset
    embedding_offset = input_id * embedding_dim
    
    # Load embedding vector
    offsets = embedding_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (num_embeddings * embedding_dim)
    
    embedding = tl.load(embedding_weight_ptr + offsets, mask=mask, other=0.0)
    
    # Store result
    output_pos = input_pos * embedding_dim
    output_offsets = output_pos + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < (batch_size * seq_len * embedding_dim)
    
    tl.store(output_ptr + output_offsets, embedding, mask=output_mask)

@torch.fx.wrap  
def optimized_embedding_lookup(input_ids, embedding_weight):
    """Optimized embedding lookup using Triton"""
    input_batch, input_seq_len = input_ids.shape
    num_embeddings, embedding_dim = embedding_weight.shape
    
    # Create output tensor
    output = torch.empty((input_batch, input_seq_len, embedding_dim), 
                        dtype=embedding_weight.dtype, 
                        device=embedding_weight.device)
    
    # Optimal block size for embeddings
    BLOCK_SIZE = min(128, embedding_dim)
    
    # Calculate grid size
    total_elements = input_batch * input_seq_len
    grid_size = (total_elements + 127) // 128
    
    # Launch kernel
    embedding_lookup_kernel[grid_size](
        input_ids_ptr=input_ids,
        embedding_weight_ptr=embedding_weight,
        output_ptr=output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        batch_size=input_batch,
        seq_len=input_seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized embedding function"""
    return optimized_embedding_lookup