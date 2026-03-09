import torch
import triton
import triton.language as tl

def pattern(in_4, in_6, in_5, tmp_3):
    # Pattern: in_4 + in_6 followed by embedding and add
    tmp_4 = in_4 + in_6
    tmp_5 = torch.nn.functional.embedding(in_5, tmp_3, 1, None, 2.0, False, False)
    tmp_4 += tmp_5
    return tmp_4

def replacement_args(in_4, in_6, in_5, tmp_3):
    return (in_4, in_6, in_5, tmp_3)

@triton.jit
def fused_embedding_add_kernel(
    x_ptr,          # inputs_embeds
    y_ptr,          # token_type_embeddings  
    pos_ids_ptr,    # position_ids
    embed_weight_ptr, # position_embeddings_parameters_weight
    out_ptr,        # fused output
    batch_size,
    seq_len,
    hidden_size,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate offsets for this program
    batch_offset = (block_start // seq_len) % batch_size
    seq_offset = (block_start % seq_len)
    
    mask = (block_start < batch_size * seq_len) & (batch_offset < batch_size) & (seq_offset < seq_len)
    
    if mask:
        # Load inputs_embeds and token_type_embeddings
        x_offset = batch_offset * seq_len * hidden_size + seq_offset * hidden_size
        y_offset = batch_offset * seq_len * hidden_size + seq_offset * hidden_size
        
        x_data = tl.load(x_ptr + x_offset, mask=mask)
        y_data = tl.load(y_ptr + y_offset, mask=mask, other=0.0)
        
        # Load position_ids
        pos_offset = batch_offset * seq_len + seq_offset
        pos_id = tl.load(pos_ids_ptr + pos_offset, mask=mask, other=0)
        
        # Compute embedding lookup using offset calculation
        embed_offset = pos_id * hidden_size
        embed_data = tl.load(embed_weight_ptr + embed_offset, mask=embed_offset < vocab_size * hidden_size, other=0.0)
        
        # Fused operation: ( inputs_embeds + token_type_embeddings ) + embedding(output)
        result = (x_data + y_data) + embed_data
        
        # Store result
        out_offset = batch_offset * seq_len * hidden_size + seq_offset * hidden_size
        tl.store(out_ptr + out_offset, result, mask=mask)

@torch.fx.wrap
def fused_embedding_addition(inputs_embeds, token_type_embeddings, position_ids, position_embeddings_weight):
    batch_size = inputs_embeds.shape[0]
    seq_len = inputs_embeds.shape[1] 
    hidden_size = inputs_embeds.shape[2]
    vocab_size = position_embeddings_weight.shape[0]
    
    output = torch.empty_like(inputs_embeds)
    
    BLOCK_SIZE = 1024
    grid_size = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_embedding_add_kernel[grid_size](
        inputs_embeds,
        token_type_embeddings,
        position_ids, 
        position_embeddings_weight,
        output,
        batch_size,
        seq_len,
        hidden_size,
        vocab_size,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_addition