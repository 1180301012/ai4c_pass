import torch

def pattern(tensor):
    """Match any redundant long() conversion on int64 tensors"""
    return tensor.long()

def replacement_args(tensor):
    """Extract arguments for replacement function"""
    return (tensor,)

def replacement_func():
    """Return a function that performs identity operation (no-op) when dtype is already int64"""
    def identity_long(x):
        # Check if tensor is already int64 - if so, avoid redundant conversion
        if x.dtype == torch.int64:
            return x  # Return as-is for int64 (avoid redundant conversion)
        else:
            # Perform actual conversion for other dtypes
            return x.long()
    
    return identity_long