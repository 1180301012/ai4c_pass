import torch
from torch import device
import triton
import triton.language as tl

@triton.jit
def embedding_index_kernel_128(
    position_ids_ptr,
    weight_ptr,
    output_ptr,
    num_embeddings,
    embedding_dim,
    seq_len,
    block_size: tl.constexpr,
):
    # Each program handles one position in the sequence
    pid = tl.program_id(0)
    
    # Ensure we don't go out of bounds
    if pid >= seq_len:
        return
    
    # Compute the final index for this position
    # Original computation: (in_1 - arange(128) + 2048 - 1)
    final_index = (tl.load(position_ids_ptr + pid) - pid + 2047)
    
    # Ensure index is within bounds
    final_index = tl.maximum(final_index, 0)
    final_index = tl.minimum(final_index, num_embeddings - 1)
    
    # Load the embedding vector
    offset = final_index * embedding_dim
    mask = tl.arange(0, block_size) < embedding_dim
    weight_data = tl.load(weight_ptr + offset + tl.arange(0, block_size), mask=mask, other=0.0)
    
    # Store the result
    tl.store(output_ptr + pid * embedding_dim + tl.arange(0, block_size), weight_data, mask=mask)

@torch.fx.wrap
def optimized_embedding_lookup_128(position_ids, weight):
    # Variable sequence length based on input
    seq_len = position_ids.shape[0]
    num_embeddings, embedding_dim = weight.shape
    
    # Use optimal block size for embedding dimension
    BLOCK_SIZE = min(embedding_dim, 128)  # Typical embedding dims are 64, 128, 256, etc.
    
    # Create output tensor
    output = torch.empty((seq_len, embedding_dim), dtype=torch.float32, device=weight.device)
    
    # Launch kernel
    num_programs = seq_len
    embedding_index_kernel_128[(num_programs,)](
        position_ids_ptr=position_ids,
        weight_ptr=weight,
        output_ptr=output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        block_size=BLOCK_SIZE,
    )
    
    return output

def pattern(tmp_0, in_1):
    # Match the exact computation for seq_len=128
    tmp_1 = torch.arange(128, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = in_1 - tmp_2
    tmp_4 = tmp_3 + 2048
    tmp_5 = tmp_4 - 1
    tmp_6 = torch.nn.functional.embedding(tmp_5, tmp_0, None, None, 2.0, False, False)
    tmp_7 = tmp_6.to(dtype=torch.float32)
    return tmp_7

def replacement_args(tmp_0, in_1):
    return (tmp_0, in_1)

def replacement_func():
    return optimized_embedding_lookup_128