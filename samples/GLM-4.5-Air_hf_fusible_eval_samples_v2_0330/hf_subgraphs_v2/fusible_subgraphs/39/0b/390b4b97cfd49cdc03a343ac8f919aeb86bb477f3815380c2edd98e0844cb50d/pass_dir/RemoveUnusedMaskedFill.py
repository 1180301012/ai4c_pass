import torch
import triton
import triton.language as tl

# Pattern matching function - entire computation sequence but skipping masked fill
def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    max_1 = tmp_6.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return (tmp_13, tmp_6)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized version - skip the masked fill operation entirely
def optimized_computation(in_0, in_1):
    # Skip masked fill since tmp_4 is never used
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_6 = tmp_2.unsqueeze(0).expand(3, -1, -1)
    
    # Continue with max operations
    max_1 = tmp_6.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    
    return (tmp_13, tmp_6)

# Replacement function
def replacement_func():
    return optimized_computation