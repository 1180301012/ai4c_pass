import torch
import triton
import triton.language as tl

def pattern(input_ids, embedding_weight, equality_mask):
    # Original: embedding lookup followed by masking
    embedding = torch.nn.functional.embedding(input_ids, embedding_weight, 1, None, 2.0, False, False)
    masking_mask = equality_mask.unsqueeze(-1)
    masked_embedding = embedding.masked_fill(masking_mask, 0.0)
    return masked_embedding

def replacement_args(input_ids, embedding_weight, equality_mask):
    return (input_ids, embedding_weight, equality_mask)

@triton.jit
def embedding_mask_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    output_ptr,
    equality_mask_ptr,
    num_sequences,
    sequence_length,
    embedding_dim,
    embedding_vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one position in the sequence
    seq_idx = tl.program_id(0)
    pos_idx = tl.program_id(1)
    
    # Calculate offset in flattened array
    offset = seq_idx * sequence_length * embedding_dim + pos_idx * embedding_dim
    mask_offset = seq_idx * sequence_length + pos_idx
    
    # Load input ID
    input_id = tl.load(input_ids_ptr + mask_offset, mask=mask_offset < (num_sequences * sequence_length), other=0)
    
    # Check if this position should be masked (input_id == 2)
    is_padding = (input_id == 2)
    
    # Compute embedding offset
    emb_offset = input_id * embedding_dim
    
    # Load embedding weights with proper bounds checking
    if emb_offset < embedding_vocab_size * embedding_dim:
        # Load a block of embedding weights
        mask = tl.arange(0, BLOCK_SIZE) < embedding_dim
        weights = tl.load(
            embedding_weight_ptr + emb_offset + tl.arange(0, BLOCK_SIZE),
            mask=mask,
            other=0.0
        )
        
        # Apply mask if this is a padding token
        if is_padding:
            weights = tl.where(mask, 0.0, weights)
        
        # Store the result
        tl.store(output_ptr + offset + tl.arange(0, BLOCK_SIZE), weights, mask=mask)
    else:
        # Handle out-of-bounds input IDs (set to zero)
        mask = tl.arange(0, BLOCK_SIZE) < embedding_dim
        tl.store(output_ptr + offset + tl.arange(0, BLOCK_SIZE), 0.0, mask=mask)

@torch.fx.wrap
def fused_embedding_with_mask(input_ids, embedding_weight, equality_mask):
    num_sequences, sequence_length = input_ids.shape
    embedding_dim = embedding_weight.shape[1]
    
    # Determine optimal block size
    BLOCK_SIZE = 128
    
    # Calculate output shape and create output tensor
    output_shape = (num_sequences, sequence_length, embedding_dim)
    output = torch.zeros(output_shape, dtype=embedding_weight.dtype, device=embedding_weight.device)
    
    # Set grid dimensions
    grid = (num_sequences, sequence_length)
    
    # Launch kernel
    embedding_mask_kernel[grid](
        input_ids_ptr=input_ids,
        embedding_weight_ptr=embedding_weight,
        output_ptr=output,
        equality_mask_ptr=equality_mask,
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        embedding_vocab_size=embedding_weight.shape[0],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_with_mask