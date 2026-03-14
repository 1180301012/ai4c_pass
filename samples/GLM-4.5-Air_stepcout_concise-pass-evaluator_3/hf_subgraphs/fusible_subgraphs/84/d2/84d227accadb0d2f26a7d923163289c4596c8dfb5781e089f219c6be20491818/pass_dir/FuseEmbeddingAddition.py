import torch
import triton
import triton.language as tl

def pattern(input_embeds, token_type_embeds, position_ids, weight_pos_embed):
    # tmp_4 = in_4 + in_6
    tmp_4 = input_embeds + token_type_embeds
    # tmp_5 = torch.nn.functional.embedding(in_5, tmp_3, 1, None, 2.0, False, False)
    tmp_5 = torch.nn.functional.embedding(position_ids, weight_pos_embed, 1, None, 2.0, False, False)
    # tmp_4 += tmp_5
    tmp_4 += tmp_5
    return tmp_4

def replacement_args(input_embeds, token_type_embeds, position_ids, weight_pos_embed):
    return (input_embeds, token_type_embeds, position_ids, weight_pos_embed)

@triton.jit
def fused_embedding_add_kernel(
    input_embeds_ptr,
    token_type_embeds_ptr,
    weight_pos_embed_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_size,
    vocab_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one batch and one sequence position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate offset for current batch and position
    batch_offset = batch_idx * seq_len * hidden_size
    pos_offset = seq_idx * hidden_size
    
    # Load input embeddings and token type embeddings
    input_offset = batch_offset + pos_offset
    token_type_offset = batch_offset + pos_offset
    
    input_embeds = tl.load(input_embeds_ptr + input_offset)
    token_type_embeds = tl.load(token_type_embeds_ptr + token_type_offset)
    
    # Initial addition
    intermediate = input_embeds + token_type_embeds
    
    # Load position embedding using position_ids
    position_offset = batch_idx * seq_len + seq_idx
    position_id = tl.load(weight_pos_embed_ptr + position_offset * 4)  # Assuming position_ids is int64, stored as int32 in tensor
    
    # Calculate position embedding offset
    pos_embed_offset = position_id * hidden_size
    
    # Load position embedding
    pos_embed = tl.load(weight_pos_embed_ptr + vocab_size * 4 + pos_embed_offset)
    
    # Final addition
    output = intermediate + pos_embed
    
    # Store result
    output_offset = batch_offset + pos_offset
    tl.store(output_ptr + output_offset, output)

@torch.fx.wrap
def fused_embedding_addition(input_embeds, token_type_embeds, position_ids, weight_pos_embed):
    batch_size, seq_len, hidden_size = input_embeds.shape
    vocab_size = weight_pos_embed.shape[0]
    
    # Create output tensor
    output = torch.empty_like(input_embeds)
    
    # Determine grid size
    batch_programs = batch_size
    seq_programs = seq_len
    
    # Launch kernel with optimized block sizes
    fused_embedding_add_kernel[(batch_programs, seq_programs)](
        input_embeds_ptr=input_embeds,
        token_type_embeds_ptr=token_type_embeds,
        weight_pos_embed_ptr=weight_pos_embed,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
    )
    
    return output

def replacement_func():
    return fused_embedding_addition