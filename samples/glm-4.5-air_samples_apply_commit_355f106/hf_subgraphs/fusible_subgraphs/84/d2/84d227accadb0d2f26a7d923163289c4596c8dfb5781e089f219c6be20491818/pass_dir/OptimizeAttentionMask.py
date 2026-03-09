import torch

def pattern(attention_mask, expand_shape):
    """Pattern to match attention mask view + expand operations"""
    # tmp_9 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    # tmp_10 = tmp_9.expand(expand_shape)
    view_result = attention_mask[slice(None, None, None), None, None, slice(None, None, None)]
    expand_result = view_result.expand(expand_shape)
    return expand_result

def replacement_args(attention_mask, expand_shape):
    """Return attention mask and expand shape for optimization"""
    return (attention_mask, expand_shape)

@torch.fx.wrap
def optimized_attention_mask(attention_mask, expand_shape):
    """Optimized attention mask expansion using direct reshape/copy operations"""
    # Original: attention_mask[1, 64] -> [1, 1, 1, 64] -> [final_shape]
    # We can optimize this by directly creating the target shape with broadcast assignment
    if attention_mask.dim() != 2:
        # Fallback to original implementation for unexpected shapes
        view_result = attention_mask[slice(None, None, None), None, None, slice(None, None, None)]
        return view_result.expand(expand_shape)
    
    # Get original dimensions
    orig_batch, orig_seq = attention_mask.shape
    
    # Instead of view then expand, create the final tensor directly with broadcast
    # This saves one intermediate tensor allocation
    if expand_shape[0] == orig_batch and expand_shape[-1] == orig_seq:
        # Create result tensor in the target shape
        result = torch.zeros(expand_shape, dtype=attention_mask.dtype, device=attention_mask.device)
        
        # Assign the attention mask to the appropriate dimensions
        # For shape [B, 1, S, S_orig], we assign to [:, :, :, :orig_seq]
        if expand_shape[1] == 1 and expand_shape[2] == expand_shape[-1]:
            # Pattern: attention_mask[B, S_orig] -> result[B, 1, S, S_orig]
            # We need to broadcast the assignment
            result[:, :, :expand_shape[-1], :orig_seq] = attention_mask.unsqueeze(1).unsqueeze(2)
        elif expand_shape[1] == 1 and expand_shape[2] != expand_shape[-1]:
            # Pattern: result[B, 1, S, S] where S != S_orig
            # This is more complex, so we fall back to simple expand
            view_result = attention_mask[slice(None, None, None), None, None, slice(None, None, None)]
            return view_result.expand(expand_shape)
        else:
            # Fallback to original for unexpected patterns
            view_result = attention_mask[slice(None, None, None), None, None, slice(None, None, None)]
            return view_result.expand(expand_shape)
            
        return result
    else:
        # Fallback to original implementation if shape doesn't match expected pattern
        view_result = attention_mask[slice(None, None, None), None, None, slice(None, None, None)]
        return view_result.expand(expand_shape)

def replacement_func():
    return optimized_attention_mask