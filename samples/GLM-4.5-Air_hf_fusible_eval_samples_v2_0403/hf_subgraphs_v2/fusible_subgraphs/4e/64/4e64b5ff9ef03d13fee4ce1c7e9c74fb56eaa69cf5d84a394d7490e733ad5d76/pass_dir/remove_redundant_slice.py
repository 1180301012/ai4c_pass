import torch

# Pattern: Slice operation that takes first 64 elements along last dimension
def pattern(tmp_3):
    return tmp_3[(Ellipsis, slice(None, 64, None))]

# Extract arguments for replacement
def replacement_args(tmp_3):
    return (tmp_3,)

# Optimized implementation that handles redundant slice
@torch.fx.wrap
def remove_redundant_slice(input_tensor):
    """
    Remove redundant slice operation when the tensor already has <= 64 elements
    in the last dimension. This optimization avoids unnecessary slicing overhead.
    """
    if input_tensor.shape[-1] <= 64:
        # Slice is redundant - return the original tensor directly
        # This avoids both the slice operation and the function call overhead
        return input_tensor
    else:
        # Perform the actual slice if needed
        return input_tensor[..., :64]

def replacement_func():
    return remove_redundant_slice