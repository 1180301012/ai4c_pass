import torch
import triton
import triton.language as tl

@triton.jit
def fused_embedding_add_kernel(
    word_ids_ptr, token_type_ids_ptr, position_ids_ptr,
    word_emb_ptr, token_type_emb_ptr, position_emb_ptr,
    out_ptr, residual_ptr,
    word_ids_stride, token_type_ids_stride, position_ids_stride,
    batch_size, seq_len, hidden_dim,
    word_num_embeddings, word_emb_stride,
    token_type_num_embeddings, token_type_emb_stride,
    position_num_embeddings, position_emb_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    num_elements = batch_size * seq_len
    
    # Calculate which sequence position this program handles
    seq_idx = pid % seq_len
    batch_idx = pid // seq_len
    
    if pid >= num_elements:
        return
    
    # Calculate offsets for ids
    word_id_offset = batch_idx * word_ids_stride + seq_idx
    token_type_id_offset = batch_idx * token_type_ids_stride + seq_idx
    position_id_offset = seq_idx  # position_ids is [1, seq_len]
    
    # Load indices
    word_idx = tl.load(word_ids_ptr + word_id_offset).to(tl.int32)
    token_type_idx = tl.load(token_type_ids_ptr + token_type_id_offset).to(tl.int32)
    position_idx = tl.load(position_ids_ptr + position_id_offset).to(tl.int32)
    
    # Load embeddings and accumulate
    result = tl.zeros((hidden_dim,), dtype=tl.float32)
    
    # Word embedding
    word_row_offset = word_idx * word_emb_stride + tl.arange(0, hidden_dim)
    word_mask = word_row_offset < word_idx * word_emb_stride + hidden_dim
    word_emb = tl.load(word_emb_ptr + word_row_offset, mask=word_mask, other=0.0)
    result = result + word_emb
    
    # Token type embedding
    token_type_row_offset = token_type_idx * token_type_emb_stride + tl.arange(0, hidden_dim)
    token_type_mask = token_type_row_offset < token_type_idx * token_type_emb_stride + hidden_dim
    token_type_emb = tl.load(token_type_emb_ptr + token_type_row_offset, mask=token_type_mask, other=0.0)
    result = result + token_type_emb
    
    # Position embedding
    position_row_offset = position_idx * position_emb_stride + tl.arange(0, hidden_dim)
    position_mask = position_row_offset < position_idx * position_emb_stride + hidden_dim
    position_emb = tl.load(position_emb_ptr + position_row_offset, mask=position_mask, other=0.0)
    result = result + position_emb
    
    # Store result
    out_offset = pid * hidden_dim + tl.arange(0, hidden_dim)
    out_mask = out_offset < num_elements * hidden_dim
    tl.store(out_ptr + out_offset, result, mask=out_mask)
    tl.store(residual_ptr + out_offset, result, mask=out_mask)


@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements, hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute mean
    pid = tl.program_id(0)
    row_offset = pid * hidden_dim
    
    # Load this row
    row_offset_range = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = row_offset_range < (pid + 1) * hidden_dim
    
    # For layer norm, process BLOCK_SIZE elements at a time
    x = tl.load(x_ptr + row_offset_range, mask=mask, other=0.0)
    
    # Compute sum
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / hidden_dim
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_dim
    inv_var = tl.rsqrt(var + eps)
    
    # Normalize
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    out = (x_centered * inv_var) * w + b
    tl.store(out_ptr + row_offset_range, out, mask=mask)


@torch.fx.wrap
def fused_embedding_add_wrapper(word_ids, token_type_ids, position_ids,
                                 word_emb_table, token_type_emb_table, position_emb_table):
    """
    Fused embedding lookup and addition kernel.
    Performs: word_emb + token_type_emb + position_emb
    """
    batch_size, seq_len = word_ids.shape
    hidden_dim = word_emb_table.shape[1]
    
    num_elements = batch_size * seq_len
    
    # Allocate output tensors
    residual = torch.empty((batch_size, seq_len, hidden_dim), 
                           dtype=word_emb_table.dtype, device=word_emb_table.device)
    
    # Launch kernel for embedding lookups and addition
    BLOCK_SIZE = 128
    num_programs = num_elements
    
    # Handle contiguous memory layouts for embedding tables
    word_emb_ptr = word_emb_table.as_strided(
        (word_emb_table.shape[0], hidden_dim),
        (hidden_dim, 1)  # row-major layout
    ).data_ptr()
    token_type_emb_ptr = token_type_emb_table.as_strided(
        (token_type_emb_table.shape[0], hidden_dim),
        (hidden_dim, 1)
    ).data_ptr()
    position_emb_ptr = position_emb_table.as_strided(
        (position_emb_table.shape[0], hidden_dim),
        (hidden_dim, 1)
    ).data_ptr()
    
    fused_embedding_add_kernel[(num_programs,)](
        word_ids, token_type_ids, position_ids,
        word_emb_ptr, token_type_emb_ptr, position_emb_ptr,
        residual, residual,  # Use residual as output
        word_ids.stride(0), token_type_ids.stride(0), position_ids.stride(0),
        batch_size, seq_len, hidden_dim,
        word_emb_table.shape[0], hidden_dim,
        token_type_emb_table.shape[0], hidden_dim,
        position_emb_table.shape[0], hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return residual


@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias, eps=1e-12):
    """
    Layer normalization kernel.
    """
    batch_size, seq_len, hidden_dim = x.shape
    num_elements = batch_size * seq_len
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    
    layer_norm_kernel[(num_elements,)](
        x, weight, bias, out,
        num_elements * hidden_dim, hidden_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Match the embedding + add + dropout + layer_norm pattern.
    
    The pattern matches:
    1. Three embedding lookups (word, token_type, position)
    2. Two add operations to combine embeddings
    3. Dropout
    4. Layer normalization
    
    Returns: (dropout_output, layer_norm_output)
    """
    # Embedding lookups with proper padding_idx arguments
    # First embedding: word_embeddings (padding_idx=0)
    tmp_7 = torch.nn.functional.embedding(in_0, in_3, 0, None, 2.0, False, False)
    # Second embedding: token_type_embeddings
    tmp_8 = torch.nn.functional.embedding(in_6, in_2, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    # Third embedding: position_embeddings
    tmp_10 = torch.nn.functional.embedding(in_7, in_1, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_11 = tmp_9
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (in_3.shape[1],), in_5, in_4, 1e-12)
    
    return tmp_12, tmp_13


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Extract arguments needed for the fused kernel.
    in_0: word_ids
    in_1: position_embeddings
    in_2: token_type_embeddings
    in_3: word_embeddings
    in_4: layer_norm bias
    in_5: layer_norm weight
    in_6: token_type_ids
    in_7: position_ids
    """
    return (in_0, in_6, in_7, in_3, in_2, in_1, in_4, in_5)


def replacement_func():
    """
    Returns the replacement function that implements the fused operation.
    """
    def compute_fused_embeddings_dropout_layernorm(word_ids, token_type_ids, position_ids,
                                                    word_emb_table, token_type_emb_table, position_emb_table,
                                                    ln_bias, ln_weight):
        """
        Fused implementation of embedding lookups + addition + dropout + layer_norm.
        
        Args:
            word_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            position_ids: [1, seq_len]
            word_emb_table: [vocab_size, hidden_dim]
            token_type_emb_table: [num_types, hidden_dim]
            position_emb_table: [max_positions, hidden_dim]
            ln_bias: [hidden_dim]
            ln_weight: [hidden_dim]
        
        Returns:
            dropout_output: Same as residual after dropout (dropout is pass-through)
            layer_norm_output: Layer normalized output
        """
        batch_size, seq_len = word_ids.shape
        hidden_dim = word_emb_table.shape[1]
        
        # Fused embedding lookup and addition
        residual = fused_embedding_add_wrapper(word_ids, token_type_ids, position_ids, 
                                               word_emb_table, token_type_emb_table, position_emb_table)
        
        # Dropout is pass-through during inference (scale=1.0)
        dropout_output = residual
        
        # Layer normalization
        layer_norm_output = layer_norm_wrapper(residual, ln_weight, ln_bias, eps=1e-12)
        
        return dropout_output, layer_norm_output
    
    return compute_fused_embeddings_dropout_layernorm