import torch
import triton
import triton.language as tl

from pass_dir.shared_kernel import shared_embedding_ln_dispatcher

# This is a variant pass for graphs where in_12 = bbox and in_13 = extended_attention_mask
# (instead of the standard in_12 = extended_attention_mask and in_13 = zeros)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
    """
    Match the embedding computation pattern for variant where:
    - in_12 = bbox (shape [1, seq_len, 4])
    - in_13 = extended_attention_mask (shape [1, 1, 1, seq_len])
    
    This is used in graphs like float16/9 and bfloat16/9
    """
    # Compute mask - note in_13 is the attention mask here
    tmp_12 = in_13.to(dtype=torch.float32)
    tmp_13 = 1.0 - tmp_12
    tmp_14 = tmp_13 * -3.4028234663852886e+38
    
    # Position IDs slicing - using dynamic slice that matches any sequence length
    seq_end = in_2.shape[1]
    tmp_15 = in_2[slice(None, None, None), slice(None, seq_end, None)]
    
    # Embedding lookups
    tmp_16 = torch.nn.functional.embedding(in_0, in_9, 0, None, 2.0, False, False)
    tmp_17 = torch.nn.functional.embedding(tmp_15, in_6, None, None, 2.0, False, False)
    
    # Bbox embeddings - in_12 is bbox here (shape [1, seq_len, 4])
    tmp_18 = in_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_19 = torch.nn.functional.embedding(tmp_18, in_10, None, None, 2.0, False, False)
    
    tmp_20 = in_12[slice(None, None, None), slice(None, None, None), 1]
    tmp_21 = torch.nn.functional.embedding(tmp_20, in_11, None, None, 2.0, False, False)
    
    tmp_22 = in_12[slice(None, None, None), slice(None, None, None), 2]
    tmp_23 = torch.nn.functional.embedding(tmp_22, in_10, None, None, 2.0, False, False)
    
    tmp_24 = in_12[slice(None, None, None), slice(None, None, None), 3]
    tmp_25 = torch.nn.functional.embedding(tmp_24, in_11, None, None, 2.0, False, False)
    
    # Bbox difference embeddings
    tmp_26 = in_12[slice(None, None, None), slice(None, None, None), 3]
    tmp_27 = in_12[slice(None, None, None), slice(None, None, None), 1]
    tmp_28 = tmp_26 - tmp_27
    tmp_29 = torch.nn.functional.embedding(tmp_28, in_5, None, None, 2.0, False, False)
    
    tmp_30 = in_12[slice(None, None, None), slice(None, None, None), 2]
    tmp_31 = in_12[slice(None, None, None), slice(None, None, None), 0]
    tmp_32 = tmp_30 - tmp_31
    tmp_33 = torch.nn.functional.embedding(tmp_32, in_8, None, None, 2.0, False, False)
    
    # Token type embedding
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
    
    # LayerNorm and Dropout
    tmp_43 = torch.nn.functional.layer_norm(tmp_42, (768,), in_4, in_3, 1e-12)
    tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False)
    
    return tmp_44, tmp_14


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
    """Extract arguments for the fused kernel variant."""
    # Determine sequence length from position_ids
    seq_len = in_2.shape[1]
    # Note: in this variant, in_12 is bbox and in_13 is mask
    # For the dispatcher: mask_or_bbox = in_13 (mask), other_tensor = in_12 (bbox)
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, "FuseEmbeddingLayerNormVariant", seq_len)


def replacement_func():
    # Use the shared dispatcher with routing
    return shared_embedding_ln_dispatcher