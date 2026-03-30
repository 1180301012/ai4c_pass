import torch

def pattern(scaled_activation, attention_scores, attention_mask):
    """
    Pattern: Multi-addition with scaled activation, attention scores, and attention mask
    Original sequence:
        tmp_11 = scaled_activation.unsqueeze(0)
        tmp_12 = attention_scores + tmp_11
        tmp_13 = tmp_12.view(1, 64, -1, 64, 64)  # features varies between graphs
        tmp_14 = attention_mask.unsqueeze(1)
        tmp_15 = tmp_14.unsqueeze(0)
        tmp_16 = tmp_13 + tmp_15
        tmp_17 = attention_mask.unsqueeze(1) 
        tmp_18 = tmp_17.unsqueeze(0)
        tmp_19 = tmp_16 + tmp_18
    """
    # Apply the exact sequence from original
    tmp_11 = scaled_activation.unsqueeze(0)
    tmp_12 = attention_scores + tmp_11
    tmp_13 = tmp_12.view(1, 64, -1, 64, 64)  # -1 will be features dimension
    tmp_14 = attention_mask.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)  
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = attention_mask.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    
    return tmp_19

def replacement_args(scaled_activation, attention_scores, attention_mask):
    return (scaled_activation, attention_scores, attention_mask)

def replacement_func():
    def optimized_multi_addition(scaled_activation, attention_scores, attention_mask):
        """Optimized version that reduces intermediate tensor creation"""
        features = scaled_activation.shape[-1]
        
        # The optimization is to avoid recreating the same mask expansion twice
        # Create mask expansion once and reuse it
        mask_expanded = attention_mask.unsqueeze(1).unsqueeze(0)
        
        # Combine operations to reduce memory overhead
        result = attention_scores + scaled_activation.unsqueeze(0) + mask_expanded + mask_expanded
        
        # Reshape to match expected output format
        result = result.view(1, 64, features, 64, 64)
        
        return result
    
    return optimized_multi_addition