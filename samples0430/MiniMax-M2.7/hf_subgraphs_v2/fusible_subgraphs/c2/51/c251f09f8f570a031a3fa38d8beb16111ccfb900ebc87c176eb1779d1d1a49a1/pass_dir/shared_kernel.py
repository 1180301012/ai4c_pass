import torch
import triton
import triton.language as tl

# Shared kernel implementations for LayoutLM embedding fusion
# Used by both FuseEmbeddingLayerNorm and FuseEmbeddingLayerNormVariant passes


@torch.fx.wrap
def shared_embedding_ln_dispatcher(
    input_ids, token_type_ids, position_ids,
    bbox, word_emb, position_emb, token_type_emb,
    x_emb, y_emb, w_emb, h_emb,
    ln_weight, ln_bias, mask_or_bbox, other_tensor,
    route, seq_len
):
    """
    Shared dispatcher that routes to the correct implementation based on route string.
    
    Route "FuseEmbeddingLayerNorm": 
        - mask_or_bbox = extended_attention_mask (in_12)
        - other_tensor = zeros (in_13)
    
    Route "FuseEmbeddingLayerNormVariant":
        - mask_or_bbox = extended_attention_mask (in_13)
        - other_tensor = bbox (in_12)
    """
    batch_size = input_ids.shape[0]
    embed_dim = 768
    
    # The output tensor
    output = torch.empty((batch_size, seq_len, embed_dim), dtype=word_emb.dtype, device=word_emb.device)
    
    # Determine which tensor is bbox based on route
    if route == "FuseEmbeddingLayerNorm":
        # Standard case: bbox is the other_tensor (in_13 in original, passed as other_tensor here)
        # mask_or_bbox is the attention mask
        bbox_tensor = other_tensor
    else:
        # Variant case: bbox is mask_or_bbox (in_12 in original)
        bbox_tensor = mask_or_bbox
    
    # Create bbox index tensors
    bbox_x = bbox_tensor[:, :, 0]
    bbox_y = bbox_tensor[:, :, 1]
    bbox_w = bbox_tensor[:, :, 2]
    bbox_h = bbox_tensor[:, :, 3]
    
    # Accumulate embeddings
    output = torch.zeros_like(output)
    
    # Word embedding
    output += torch.nn.functional.embedding(input_ids, word_emb, 0, None, 2.0, False, False)
    # Position embedding  
    output += torch.nn.functional.embedding(position_ids[:, :seq_len], position_emb, None, None, 2.0, False, False)
    # Token type embedding
    output += torch.nn.functional.embedding(token_type_ids, token_type_emb, None, None, 2.0, False, False)
    # Bbox x embedding
    output += torch.nn.functional.embedding(bbox_x, x_emb, None, None, 2.0, False, False)
    # Bbox y embedding
    output += torch.nn.functional.embedding(bbox_y, y_emb, None, None, 2.0, False, False)
    # Bbox w embedding (need differences)
    bbox_w_diff = bbox_tensor[:, :, 2] - bbox_tensor[:, :, 0]
    output += torch.nn.functional.embedding(bbox_w_diff, w_emb, None, None, 2.0, False, False)
    # Bbox h embedding (need differences)
    bbox_h_diff = bbox_tensor[:, :, 3] - bbox_tensor[:, :, 1]
    output += torch.nn.functional.embedding(bbox_h_diff, h_emb, None, None, 2.0, False, False)
    
    # Apply LayerNorm
    output = torch.nn.functional.layer_norm(output, (embed_dim,), ln_weight, ln_bias, 1e-12)
    
    # Apply Dropout
    output = torch.nn.functional.dropout(output, 0.1, True, False)
    
    return output