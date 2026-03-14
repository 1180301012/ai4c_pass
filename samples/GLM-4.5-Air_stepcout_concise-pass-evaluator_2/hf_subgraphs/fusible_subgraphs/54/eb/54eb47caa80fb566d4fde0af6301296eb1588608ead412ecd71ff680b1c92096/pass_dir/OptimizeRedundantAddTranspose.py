import torch

def pattern(in_0, in_1):
    """
    Pattern that matches the computation with redundant addition.
    """
    tmp_0 = in_1.reshape(1, 64, -1)
    tmp_1 = in_0 + tmp_0
    tmp_2 = in_0 + tmp_0
    tmp_3 = tmp_1.transpose(0, 1)
    tmp_4 = tmp_2.transpose(0, 1)
    tmp_5 = in_0.transpose(0, 1)
    return (tmp_4, tmp_3, tmp_5)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def optimized_reshape_add_transpose_op(in_0, in_1):
    """
    Fully optimized version that eliminates all redundant computation
    and maximizes memory efficiency.
    """
    # The key insight: compute reshape + add only once, then reuse results
    # This eliminates both the redundant addition and the redundant intermediate tensors
    
    # Step 1: Compute the common reshape operation once
    tmp_0 = in_1.reshape(1, 64, -1)
    
    # Step 2: Compute the addition only once (was duplicated in original)
    tmp_1 = in_0 + tmp_0
    
    # Step 3: tmp_2 is redundant - just reuse tmp_1 (no computation)
    tmp_2 = tmp_1
    
    # Step 4: Compute the two transposes that produce identical results
    # tmp_3 and tmp_4 are identical because tmp_1 and tmp_2 are identical
    tmp_3 = tmp_1.transpose(0, 1)
    tmp_4 = tmp_2.transpose(0, 1)  # Same as tmp_3, reusing computation result
    
    # Step 5: Compute the final transpose
    tmp_5 = in_0.transpose(0, 1)
    
    # Return the tuple in the expected order
    return (tmp_4, tmp_3, tmp_5)

def replacement_func():
    return optimized_reshape_add_transpose_op