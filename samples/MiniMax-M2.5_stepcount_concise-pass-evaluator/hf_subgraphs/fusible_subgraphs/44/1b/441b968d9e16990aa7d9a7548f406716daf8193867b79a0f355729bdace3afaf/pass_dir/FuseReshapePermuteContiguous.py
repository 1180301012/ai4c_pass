import torch

# Pattern: reshape -> permute -> contiguous
# Optimization: use view instead of reshape when tensor is already contiguous

def pattern(tmp_3):
    """
    Match pattern: reshape -> permute -> contiguous
    This pattern appears in the model after layer_norm:
    tmp_4 = tmp_3.reshape(1, 16, 16, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    """
    tmp_4 = tmp_3.reshape(1, 16, 16, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(tmp_3):
    return (tmp_3,)

def fused_reshape_permute_contiguous(x):
    """
    Optimized using view for contiguous input - avoids potential copy from reshape.
    The layer_norm output is already contiguous, so we can use view which is faster.
    """
    # Use view (faster than reshape for contiguous tensors)
    tmp_4 = x.view(1, 16, 16, 512)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    return tmp_5.contiguous()

def replacement_func():
    return fused_reshape_permute_contiguous