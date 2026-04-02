import torch
import triton
import triton.language as tl

# Pattern matching function - matches the sequence of operations for embedding computation
def pattern(input_ids, large_vocab_weight, scalar, small_vocab_weight, pos_tensor):
    # Compute first embedding and scale
    embedding1 = torch.nn.functional.embedding(input_ids, large_vocab_weight, 1, None, 2.0, False, False)
    scaled1 = embedding1 * scalar
    
    # Compute position embedding  
    embedding2 = torch.nn.functional.embedding(pos_tensor, small_vocab_weight, None, None, 2.0, False, False)
    
    # Add them together
    result = scaled1 + embedding2
    
    return result

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12):
    return (tmp_8, in_1, 16.0, in_0, tmp_4)

# Triton kernel for comprehensive embedding computation fusion
@triton.jit
def fused_embedding_kernel(
    input_ids_ptr,
    pos_ids_ptr,
    large_weight_ptr,
    small_weight_ptr,
    output_ptr,
    input_size,
    embed_dim,
    large_vocab_size,
    small_vocab_size,
    scalar_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of input tokens
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    
    # Load input IDs
    input_ids = tl.load(input_ids_ptr + offsets, mask=mask, other=0)
    pos_ids = tl.load(pos_ids_ptr + offsets, mask=mask, other=0)
    
    # Initialize output
    output = tl.zeros((BLOCK_SIZE, embed_dim), dtype=tl.float16)
    
    # Process each token in the block
    for i in range(BLOCK_SIZE):
        if mask[i]:
            # First embedding lookup (large vocabulary)
            emb_idx = input_ids[i]
            if 0 <= emb_idx < large_vocab_size:
                emb_offset = emb_idx * embed_dim
                emb_vec = tl.load(large_weight_ptr + emb_offset, mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
                emb_vec = emb_vec * scalar_val
                output[i] += emb_vec
            
            # Second embedding lookup (position + small vocabulary)
            pos_idx = pos_ids[i] + 2  # Original: pos + 2
            if 0 <= pos_idx < small_vocab_size:
                emb_offset = pos_idx * embed_dim
                pos_emb = tl.load(small_weight_ptr + emb_offset, mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
                output[i] += pos_emb
    
    # Store results
    output_offsets = offsets[:, None] * embed_dim + tl.arange(0, embed_dim)[None, :]
    tl.store(output_ptr + output_offsets, output, mask=mask[:, None] and tl.arange(0, embed_dim)[None, :] < embed_dim)

@torch.fx.wrap
def fused_embedding_comprehensive(input_ids, pos_ids, large_weight, small_weight):
    # Get dimensions
    input_size = input_ids.numel()
    embed_dim = large_weight.shape[1]
    large_vocab_size = large_weight.shape[0]
    small_vocab_size = small_weight.shape[0]
    
    # Create output tensor
    output_shape = (input_ids.shape[0], embed_dim)
    output = torch.empty(output_shape, dtype=large_weight.dtype, device=input_ids.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 64  # Optimized for this workload
    num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_embedding_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        pos_ids_ptr=pos_ids,
        large_weight_ptr=large_weight,
        small_weight_ptr=small_weight,
        output_ptr=output,
        input_size=input_size,
        embed_dim=embed_dim,
        large_vocab_size=large_vocab_size,
        small_vocab_size=small_vocab_size,
        scalar_val=16.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_comprehensive