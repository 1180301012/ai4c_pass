"""
Fuse all 8 embedding lookups and the 7 addition operations into a single optimized kernel.
This eliminates multiple kernel launches and reduces memory bandwidth by performing all
lookup operations and the addition chain in a single fused kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_embedding_add_kernel(
    # Input IDs
    input_ids_ptr, token_type_ids_ptr, position_ids_ptr,
    # Position indices from bbox (zeros tensor)
    bbox_ptr,
    # Relative position indices
    bbox_3_ptr, bbox_1_ptr, bbox_2_ptr, bbox_0_ptr,
    # Embedding tables
    word_emb_ptr, pos_emb_ptr, x_pos_emb_ptr, y_pos_emb_ptr,
    h_pos_emb_ptr, w_pos_emb_ptr, token_type_emb_ptr,
    # Output
    output_ptr,
    # Metadata
    batch_size, seq_len,
    # Strides for input_ids
    input_ids_stride0, input_ids_stride1,
    # Strides for bbox
    bbox_stride0, bbox_stride1, bbox_stride2,
    # Embedding dimension
    emb_dim: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for batching
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate offset for current position
    input_offset = batch_idx * input_ids_stride0 + seq_idx * input_ids_stride1
    
    # Calculate output offset
    out_offset = batch_idx * seq_len * emb_dim + seq_idx * emb_dim
    
    # Load word embedding index and get embedding
    word_idx = tl.load(input_ids_ptr + input_offset).to(tl.int32)
    word_emb_offset = word_idx * emb_dim
    word_emb_offsets = word_emb_offset + tl.arange(0, emb_dim)
    word_emb = tl.load(word_emb_ptr + word_emb_offsets)
    
    # Load position embedding index (from position_ids)
    pos_idx = tl.load(position_ids_ptr + seq_idx).to(tl.int32)
    pos_emb_offset = pos_idx * emb_dim
    pos_emb_offsets = pos_emb_offset + tl.arange(0, emb_dim)
    pos_emb = tl.load(pos_emb_ptr + pos_emb_offsets)
    
    # Load bbox indices for relative positions
    bbox_idx_3 = tl.load(bbox_3_ptr + batch_idx * bbox_stride0 + seq_idx * bbox_stride1).to(tl.int32)
    bbox_idx_1 = tl.load(bbox_1_ptr + batch_idx * bbox_stride0 + seq_idx * bbox_stride1).to(tl.int32)
    bbox_idx_2 = tl.load(bbox_2_ptr + batch_idx * bbox_stride0 + seq_idx * bbox_stride1).to(tl.int32)
    bbox_idx_0 = tl.load(bbox_0_ptr + batch_idx * bbox_stride0 + seq_idx * bbox_stride1).to(tl.int32)
    
    # Compute relative positions
    rel_h_idx = bbox_idx_3 - bbox_idx_1
    rel_w_idx = bbox_idx_2 - bbox_idx_0
    
    # Load token type embedding
    tt_idx = tl.load(token_type_ids_ptr + input_offset).to(tl.int32)
    tt_emb_offset = tt_idx * emb_dim
    tt_emb_offsets = tt_emb_offset + tl.arange(0, emb_dim)
    tt_emb = tl.load(token_type_emb_ptr + tt_emb_offsets)
    
    # Load h, w position embeddings
    h_emb_offset = rel_h_idx * emb_dim
    h_emb_offsets = h_emb_offset + tl.arange(0, emb_dim)
    h_emb = tl.load(h_pos_emb_ptr + h_emb_offsets)
    
    w_emb_offset = rel_w_idx * emb_dim
    w_emb_offsets = w_emb_offset + tl.arange(0, emb_dim)
    w_emb = tl.load(w_pos_emb_ptr + w_emb_offsets)
    
    # Load bbox column embeddings
    bbox_col0_offset = batch_idx * bbox_stride0 + seq_idx * bbox_stride1
    bbox_col1_offset = bbox_col0_offset + bbox_stride2
    bbox_col2_offset = bbox_col0_offset + 2 * bbox_stride2
    bbox_col3_offset = bbox_col0_offset + 3 * bbox_stride2
    
    bbox_col0_idx = tl.load(bbox_ptr + bbox_col0_offset).to(tl.int32)
    bbox_col1_idx = tl.load(bbox_ptr + bbox_col1_offset).to(tl.int32)
    bbox_col2_idx = tl.load(bbox_ptr + bbox_col2_offset).to(tl.int32)
    bbox_col3_idx = tl.load(bbox_ptr + bbox_col3_offset).to(tl.int32)
    
    # Load x, y position embeddings for each bbox column
    x0_emb_offset = bbox_col0_idx * emb_dim
    x0_emb_offsets = x0_emb_offset + tl.arange(0, emb_dim)
    x0_emb = tl.load(x_pos_emb_ptr + x0_emb_offsets)
    
    y1_emb_offset = bbox_col1_idx * emb_dim
    y1_emb_offsets = y1_emb_offset + tl.arange(0, emb_dim)
    y1_emb = tl.load(y_pos_emb_ptr + y1_emb_offsets)
    
    x2_emb_offset = bbox_col2_idx * emb_dim
    x2_emb_offsets = x2_emb_offset + tl.arange(0, emb_dim)
    x2_emb = tl.load(x_pos_emb_ptr + x2_emb_offsets)
    
    y3_emb_offset = bbox_col3_idx * emb_dim
    y3_emb_offsets = y3_emb_offset + tl.arange(0, emb_dim)
    y3_emb = tl.load(y_pos_emb_ptr + y3_emb_offsets)
    
    # Sum all embeddings
    total = word_emb + pos_emb + tt_emb + h_emb + w_emb + x0_emb + y1_emb + x2_emb + y3_emb
    
    # Store result
    out_offsets = out_offset + tl.arange(0, emb_dim)
    tl.store(output_ptr + out_offsets, total)


@torch.fx.wrap
def fused_embedding_add_wrapper(
    input_ids, token_type_ids, position_ids,
    bbox,
    word_emb, pos_emb, x_pos_emb, y_pos_emb,
    h_pos_emb, w_pos_emb, token_type_emb
):
    batch_size, seq_len = input_ids.shape
    emb_dim = word_emb.shape[1]
    
    # Prepare bbox index arrays (columns 0, 1, 2, 3 and differences)
    bbox_3 = bbox[:, :, 3].contiguous()
    bbox_1 = bbox[:, :, 1].contiguous()
    bbox_2 = bbox[:, :, 2].contiguous()
    bbox_0 = bbox[:, :, 0].contiguous()
    
    output = torch.empty((batch_size, seq_len, emb_dim), dtype=word_emb.dtype, device=word_emb.device)
    
    # Grid: (batch_size, seq_len)
    grid = (batch_size, seq_len)
    
    fused_embedding_add_kernel[grid](
        input_ids, token_type_ids, position_ids,
        bbox, bbox_3, bbox_1, bbox_2, bbox_0,
        word_emb, pos_emb, x_pos_emb, y_pos_emb,
        h_pos_emb, w_pos_emb, token_type_emb,
        output,
        batch_size, seq_len,
        input_ids.stride(0), input_ids.stride(1),
        bbox.stride(0), bbox.stride(1), bbox.stride(2),
        emb_dim,
        BLOCK_SIZE=emb_dim,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_6, in_13, in_9, in_10, in_11, in_5, in_8, in_7):
    """Match the pattern of 8 embeddings + 7 additions + layer_norm"""
    # in_0 = input_ids, in_1 = token_type_ids, in_2 = position_ids
    # in_6 = position_embeddings, in_13 = zeros/bbox
    # in_9 = word_embeddings, in_10 = x_pos_embeddings, in_11 = y_pos_embeddings
    # in_5 = h_position_embeddings, in_8 = w_position_embeddings, in_7 = token_type_embeddings
    
    # Embedding 1: word embeddings
    tmp_16 = torch.nn.functional.embedding(in_0, in_9, 0, None, 2.0, False, False)
    
    # Embedding 2: position embeddings
    tmp_15 = in_2[:, :in_2.shape[1]]
    tmp_17 = torch.nn.functional.embedding(tmp_15, in_6, None, None, 2.0, False, False)
    
    # Bbox column embeddings
    tmp_18 = in_13[:, :, 0]
    tmp_19 = torch.nn.functional.embedding(tmp_18, in_10, None, None, 2.0, False, False)
    
    tmp_20 = in_13[:, :, 1]
    tmp_21 = torch.nn.functional.embedding(tmp_20, in_11, None, None, 2.0, False, False)
    
    tmp_22 = in_13[:, :, 2]
    tmp_23 = torch.nn.functional.embedding(tmp_22, in_10, None, None, 2.0, False, False)
    
    tmp_24 = in_13[:, :, 3]
    tmp_25 = torch.nn.functional.embedding(tmp_24, in_11, None, None, 2.0, False, False)
    
    # Relative position embeddings
    tmp_26 = in_13[:, :, 3]
    tmp_27 = in_13[:, :, 1]
    tmp_28 = tmp_26 - tmp_27
    tmp_29 = torch.nn.functional.embedding(tmp_28, in_5, None, None, 2.0, False, False)
    
    tmp_30 = in_13[:, :, 2]
    tmp_31 = in_13[:, :, 0]
    tmp_32 = tmp_30 - tmp_31
    tmp_33 = torch.nn.functional.embedding(tmp_32, in_8, None, None, 2.0, False, False)
    
    # Token type embeddings
    tmp_34 = torch.nn.functional.embedding(in_1, in_7, None, None, 2.0, False, False)
    
    # Addition chain
    tmp_35 = tmp_16 + tmp_17
    tmp_36 = tmp_35 + tmp_19
    tmp_37 = tmp_36 + tmp_21
    tmp_38 = tmp_37 + tmp_23
    tmp_39 = tmp_38 + tmp_25
    tmp_40 = tmp_39 + tmp_29
    tmp_41 = tmp_40 + tmp_33
    tmp_42 = tmp_41 + tmp_34
    
    return tmp_42


def replacement_args(in_0, in_1, in_2, in_6, in_13, in_9, in_10, in_11, in_5, in_8, in_7):
    return (in_0, in_1, in_2, in_13, in_9, in_6, in_10, in_11, in_5, in_8, in_7)


def replacement_func():
    return fused_embedding_add_wrapper