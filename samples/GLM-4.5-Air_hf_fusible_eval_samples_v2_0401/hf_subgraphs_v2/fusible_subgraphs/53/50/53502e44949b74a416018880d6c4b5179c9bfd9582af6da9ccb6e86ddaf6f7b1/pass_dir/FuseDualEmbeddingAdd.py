import torch
import triton
import triton.language as tl

# Pattern matching function - matches the two embedding operations and their addition
def pattern(in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12):
    # Use input variables as they are passed
    _ = in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12
    
    # First embedding lookup
    tmp_4 = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    tmp_5 = tmp_4 * 16.0
    
    # Position tensor creation (exact match from original)
    from torch import device
    pos_tensor = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda', index=0))
    pos_tensor_expanded = pos_tensor.expand(1, -1)
    pos_tensor_offset = pos_tensor_expanded + 2
    
    # Second embedding lookup
    tmp_9 = torch.nn.functional.embedding(pos_tensor_offset, in_0, None, None, 2.0, False, False)
    
    # Addition of the two embedding results
    result = tmp_5 + tmp_9
    
    return result

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, tmp_12):
    return (in_4, in_1, 16.0, in_0, 2)

# Triton kernel for fused dual embedding
@triton.jit
def fused_embedding_kernel(
    input_ids_ptr,
    weight1_ptr,      # in_1: [64044, 256]
    weight2_ptr,      # in_0: [514, 256]
    output_ptr,
    input_ids_size,
    vocab_size1,      # 64044
    vocab_size2,      # 514
    embed_dim,        # 256
    pos_offset,       # 2
    scale1,           # 16.0
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output position
    idx = tl.program_id(0)
    
    if idx >= input_ids_size:
        return
    
    # Load input IDs
    input_ids = tl.load(input_ids_ptr + idx)
    
    # Check bounds for embedding lookups
    mask1 = (input_ids < vocab_size1) & (input_ids >= 0)
    mask2 = True  # For position embedding, we create indices synthetically
    
    # Calculate position indices for second embedding (original logic: arange + expand + add)
    pos_idx = idx + pos_offset
    
    # Load embeddings with bounds checking
    if mask1:
        # First embedding lookup
        offset1 = input_ids * embed_dim
        emb1 = tl.load(weight1_ptr + offset1, mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
        emb1 = emb1 * scale1
    else:
        emb1 = tl.zeros((embed_dim,), dtype=tl.float16)
    
    # Second embedding lookup for position
    if pos_idx < vocab_size2:
        offset2 = pos_idx * embed_dim
        emb2 = tl.load(weight2_ptr + offset2, mask=tl.arange(0, embed_dim) < embed_dim, other=0.0)
    else:
        emb2 = tl.zeros((embed_dim,), dtype=tl.float16)
    
    # Add the embeddings
    result = emb1 + emb2
    
    # Store result
    tl.store(output_ptr + idx * embed_dim, result)

@torch.fx.wrap
def fused_embedding_forward(input_ids, weight1, weight2):
    # Determine output shape
    output_shape = (input_ids.shape[0], weight1.shape[1])  # (1, 256)
    output = torch.empty(output_shape, dtype=weight1.dtype, device=input_ids.device)
    
    # Flatten input_ids for simpler processing
    input_ids_flat = input_ids.flatten()
    input_size = input_ids_flat.numel()
    embed_dim = weight1.shape[1]
    
    # Set up Triton kernel launch
    BLOCK_SIZE = 256
    num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_embedding_kernel[(num_programs,)](
        input_ids_ptr=input_ids_flat,
        weight1_ptr=weight1,
        weight2_ptr=weight2,
        output_ptr=output,
        input_ids_size=input_size,
        vocab_size1=weight1.shape[0],
        vocab_size2=weight2.shape[0],
        embed_dim=embed_dim,
        pos_offset=2,
        scale1=16.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_forward