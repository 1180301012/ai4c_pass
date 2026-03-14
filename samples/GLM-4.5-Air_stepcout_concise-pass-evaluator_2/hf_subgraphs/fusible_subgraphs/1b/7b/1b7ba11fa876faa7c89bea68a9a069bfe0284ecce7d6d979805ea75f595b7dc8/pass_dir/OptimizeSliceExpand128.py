import torch

def pattern(in_0, in_1):
    """
    Pattern that matches slice to 128 followed by expand for Mahmoud8 and bge-base models
    """
    # Create the exact operations from the observed patterns
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = tmp_1[slice(None, None, None), slice(None, 128, None)]
    tmp_1 = None
    tmp_3 = tmp_2.expand(1, 128)  # Note: expand(1, 128) is a no-op for shape [1,128]
    tmp_2 = None
    tmp_4 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_0 = None
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    def optimized_replacement(in_0, in_1):
        """Optimized implementation for 128 slice pattern"""
        # For these patterns, the expand is often a no-op, so we optimize
        
        # Step 1: Slice from second input
        sliced_tensor = in_1[:, :128]
        
        # Step 2: Since expand(1, 128) on [1, 128] is a no-op, just use the slice
        # This avoids the unnecessary expand operation
        expanded_tensor = sliced_tensor  # Skip expand when it's redundant
        
        # Step 3: Add None dimensions efficiently
        modified_input = in_0[:, None, None, :]
        
        return (expanded_tensor, modified_input)
    return optimized_replacement