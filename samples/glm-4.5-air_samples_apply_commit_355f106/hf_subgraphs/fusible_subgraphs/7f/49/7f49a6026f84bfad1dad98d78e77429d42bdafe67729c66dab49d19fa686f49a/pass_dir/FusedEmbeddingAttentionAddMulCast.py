import torch
import triton
import triton.language as tl
from torch.fx import Node

def pattern(tmp_1, tmp_3, tmp_2, tmp_0, in_4):
    """
    Pattern that matches:
    - Two embedding operations with word and position embeddings
    - Addition of the two embedding results
    - Multiplication with expanded attention mask
    - Type conversion to float32
    """
    tmp_4 = torch.nn.functional.embedding(tmp_1, tmp_3, 1, None, 2.0, False, False)
    tmp_5 = torch.nn.functional.embedding(in_4, tmp_2, 1, None, 2.0, False, False)
    tmp_6 = tmp_4 + tmp_5
    tmp_7 = tmp_0.unsqueeze(-1)
    tmp_8 = tmp_6 * tmp_7
    tmp_9 = tmp_8.to(torch.float32)
    
    # Return tuple with same structure as original model return (only tmp_9)
    return (tmp_9,)

def replacement_args(tmp_1, tmp_3, tmp_2, tmp_0, in_4):
    """
    Extract arguments needed for the fused kernel:
    - input_ids: tmp_1
    - word_embeddings: tmp_3  
    - position_embeddings: tmp_2
    - attention_mask: tmp_0
    - position_ids: in_4
    """
    return (tmp_1, tmp_3, tmp_2, tmp_0, in_4)

@triton.jit
def fused_embedding_kernel(
    # Input pointers
    input_ids_ptr,
    position_ids_ptr, 
    # Embedding weight pointers
    word_embeddings_ptr,
    position_embeddings_ptr,
    attention_mask_ptr,
    # Output pointer
    output_ptr,
    # Metadata
    input_ids_stride_0,
    input_ids_stride_1,
    position_ids_stride_0,
    position_ids_stride_1,
    word_embeddings_stride_0,
    word_embeddings_stride_1,
    position_embeddings_stride_0,
    position_embeddings_stride_1,
    attention_mask_stride_0,
    attention_mask_stride_1,
    attention_mask_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    
    # Shapes
    batch_size,
    seq_len,
    embedding_dim,
    vocab_size,
    num_positions,
    
    BLOCK_SIZE_M: tl.constexpr,  # batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # sequence dimension  
    BLOCK_SIZE_K: tl.constexpr,  # embedding dimension
):
    """
    Fused kernel that:
    1. Computes two embedding operations in parallel
    2. Adds the results
    3. Multiplies with attention mask
    4. Converts output to float32
    """
    # Program identifiers
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Create offsets for loading
    offsets_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Mask to avoid out-of-bounds access
    mask_m = offsets_m < batch_size
    mask_n = offsets_n < seq_len
    mask_k = offsets_k < embedding_dim
    
    # Load input ids and position ids
    input_ids_offset = offsets_m[:, None] * input_ids_stride_0 + offsets_n[None, :] * input_ids_stride_1
    position_ids_offset = offsets_m[:, None] * position_ids_stride_0 + offsets_n[None, :] * position_ids_stride_1
    
    input_ids = tl.load(input_ids_ptr + input_ids_offset, mask=mask_m[:, None] & mask_n[None, :], other=0)
    position_ids = tl.load(position_ids_ptr + position_ids_offset, mask=mask_m[:, None] & mask_n[None, :], other=0)
    
    # Load pre-expanded attention mask (already 3D with last dimension=1)
    attention_mask_offset = (offsets_m[:, None] * attention_mask_stride_0 + 
                           offsets_n[None, :] * attention_mask_stride_1 + 
                           offsets_k[None, None, :] * attention_mask_stride_2)
    attention_mask = tl.load(attention_mask_ptr + attention_mask_offset, 
                           mask=mask_m[:, None] & mask_n[None, :] & mask_k[None, None, :], 
                           other=1.0)
    
    # Compute word embeddings: lookup each token index in word_embeddings
    word_emb_offsets = input_ids[:, :, None] * word_embeddings_stride_0 + offsets_k[None, None, :]
    word_emb = tl.load(word_embeddings_ptr + word_emb_offsets, 
                       mask=(input_ids[:, :, None] >= 0) & (input_ids[:, :, None] < vocab_size)[:, :, None] & mask_k[None, None, :], 
                       other=0.0)
    
    # Compute position embeddings: lookup each position index in position_embeddings  
    pos_emb_offsets = position_ids[:, :, None] * position_embeddings_stride_0 + offsets_k[None, None, :]
    pos_emb = tl.load(position_embeddings_ptr + pos_emb_offsets,
                      mask=(position_ids[:, :, None] >= 0) & (position_ids[:, :, None] < num_positions)[:, :, None] & mask_k[None, None, :],
                      other=0.0)
    
    # Add embeddings
    combined_emb = word_emb + pos_emb
    
    # Multiply with attention mask
    output = combined_emb * attention_mask
    
    # Store result (already float32 due to implicit conversion in triton with attention_mask float32)
    output_offset = offsets_m[:, None] * output_stride_0 + offsets_n[None, :] * output_stride_1 + offsets_k[None, None, :]
    tl.store(output_ptr + output_offset, output, 
             mask=mask_m[:, None] & mask_n[None, :] & mask_k[None, None, :])

@torch.fx.wrap
def fused_embedding_operation(input_ids, word_embeddings, position_embeddings, attention_mask, position_ids):
    """
    Wrapper function to launch the fused Triton kernel
    """
    from typing import Tuple
    
    # Determine shapes from inputs
    batch_size, seq_len = input_ids.shape
    embedding_dim = word_embeddings.shape[1]
    vocab_size = word_embeddings.shape[0]
    num_positions = position_embeddings.shape[0]
    
    # Prepare output tensor
    output_shape = (batch_size, seq_len, embedding_dim)
    output = torch.empty(output_shape, dtype=torch.float32, device=input_ids.device)
    
    # Pre-expand attention mask for broadcasting in kernel (avoid using .unsqueeze() in Triton)
    # Convert to float32 and add last dimension for broadcasting with embeddings
    attention_mask_expanded = attention_mask.to(torch.float32).unsqueeze(-1)
    
    # Calculate strides
    input_ids_stride_0, input_ids_stride_1 = input_ids.stride()
    position_ids_stride_0, position_ids_stride_1 = position_ids.stride()
    word_embeddings_stride_0, word_embeddings_stride_1 = word_embeddings.stride()
    position_embeddings_stride_0, position_embeddings_stride_1 = position_embeddings.stride()
    attention_mask_stride_0, attention_mask_stride_1, attention_mask_stride_2 = attention_mask_expanded.stride()
    output_stride_0, output_stride_1, output_stride_2 = output.stride()
    
    # Block sizes - optimized for typical transformer dimensions
    BLOCK_SIZE_M = 8   # batch dimension
    BLOCK_SIZE_N = 64  # sequence dimension  
    BLOCK_SIZE_K = 32  # embedding dimension
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (embedding_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel
    fused_embedding_kernel[(
        grid_m,
        grid_n, 
        grid_k
    )](
        # Input pointers
        input_ids_ptr=input_ids,
        position_ids_ptr=position_ids,
        # Embedding weight pointers  
        word_embeddings_ptr=word_embeddings,
        position_embeddings_ptr=position_embeddings,
        attention_mask_ptr=attention_mask_expanded,
        # Output pointer
        output_ptr=output,
        # Strides
        input_ids_stride_0=input_ids_stride_0,
        input_ids_stride_1=input_ids_stride_1,
        position_ids_stride_0=position_ids_stride_0,
        position_ids_stride_1=position_ids_stride_1,
        word_embeddings_stride_0=word_embeddings_stride_0,
        word_embeddings_stride_1=word_embeddings_stride_1,
        position_embeddings_stride_0=position_embeddings_stride_0,
        position_embeddings_stride_1=position_embeddings_stride_1,
        attention_mask_stride_0=attention_mask_stride_0,
        attention_mask_stride_1=attention_mask_stride_1,
        attention_mask_stride_2=attention_mask_stride_2,
        output_stride_0=output_stride_0,
        output_stride_1=output_stride_1,
        output_stride_2=output_stride_2,
        # Shapes
        batch_size=batch_size,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        num_positions=num_positions,
        # Block sizes
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Return tuple to match expected pattern structure
    return (output,)

def replacement_func():
    """
    Returns the fused embedding function
    """
    return fused_embedding_operation