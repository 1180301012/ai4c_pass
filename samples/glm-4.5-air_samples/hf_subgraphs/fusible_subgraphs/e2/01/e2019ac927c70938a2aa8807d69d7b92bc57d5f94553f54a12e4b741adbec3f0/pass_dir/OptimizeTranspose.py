import torch

# Match the entire computation from the original forward function
def pattern(in_0, in_1, in_2):
    # This matches the entire computation sequence:
    # tmp_0 = in_1.exp()
    # tmp_1 = in_2 * tmp_0  
    # tmp_2 = tmp_1 + in_0
    # tmp_3 = tmp_2.t()
    # return (tmp_2, tmp_3)
    
    tmp_0 = in_1.exp()
    tmp_1 = in_2 * tmp_0
    tmp_2 = tmp_1 + in_0
    tmp_3 = tmp_2.t()
    return (tmp_2, tmp_3)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized version that eliminates intermediate variables
def optimized_forward(logit_bias, logit_scale, logits_per_text):
    # Direct computation without intermediate variables
    # This reduces memory operations and avoids cleanup overhead
    exp_scale = logit_scale.exp()
    result = logits_per_text * exp_scale + logit_bias
    return result, result.t()

def replacement_func():
    return optimized_forward