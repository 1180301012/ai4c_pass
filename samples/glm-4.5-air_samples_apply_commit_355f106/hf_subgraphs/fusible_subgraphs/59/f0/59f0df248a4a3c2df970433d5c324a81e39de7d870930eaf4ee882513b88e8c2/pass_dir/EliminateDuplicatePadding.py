import torch

def pattern(input_ids, attention_mask):
    # This simulates the duplicate computation pattern
    padding_mask1 = input_ids.__eq__(2)
    padding_count1 = padding_mask1.sum(-1).float()
    attention_sum1 = attention_mask.sum(-1).float()
    
    # This is the duplicate computation we want to eliminate
    padding_mask2 = input_ids.__eq__(2)  # Duplicate of padding_mask1
    padding_count2 = padding_mask2.sum(-1).float()  # Duplicate of padding_count1
    
    return attention_sum1, padding_count1

def replacement_args(input_ids, attention_mask):
    return (input_ids, attention_mask)

@torch.fx.wrap
def optimized_duplicate_elimination(input_ids, attention_mask):
    # Compute padding detection ONCE to eliminate duplicate
    padding_mask = input_ids.__eq__(2)
    padding_count = padding_mask.sum(-1).float()
    attention_sum = attention_mask.sum(-1).float()
    
    return attention_sum, padding_count

def replacement_func():
    return optimized_duplicate_elimination