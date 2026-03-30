import torch

# Pattern matching function - entire computation sequence with optimizations
def pattern(in_0, in_1, tmp_2, tmp_11):
    # Skip unused masked_fill operation
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    # Skip redundant device transfer since original tensors are already on GPU
    max_1 = tmp_6.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return (tmp_13, tmp_6)

# Argument extraction function
def replacement_args(in_0, in_1, tmp_2, tmp_11):
    return (in_0, in_1, tmp_2, tmp_11)

# Optimized version - comprehensive optimization
@torch.fx.wrap
def optimized_computation(in_0, in_1, tmp_2, tmp_11):
    # Note: The pattern matching extracts these at the right stages of computation
    # We're creating multiple optimization opportunities:
    # 1. Skip masked_fill (tmp_4 is unused)
    # 2. Optimize expand operation
    # 3. Optimize final arithmetic
    
    # Direct expansion without intermediate steps
    expanded = tmp_2.expand(3, -1, -1)
    
    # Max operations on expanded tensor
    max_1 = expanded.max(0, keepdim=False)[0]
    max_2 = max_1.max(-1, keepdim=True)[0]
    
    # Optimized final arithmetic: (max_2 + 1) - 9 = max_2 - 8
    final_result = max_2 - 8
    
    return (final_result, expanded)

# Replacement function
def replacement_func():
    return optimized_computation