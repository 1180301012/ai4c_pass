import torch


def pattern(in_1):
    """
    Pattern for view + transpose on in_1
    Matches: in_1.view(32, -1, 1, 64).transpose(1, 2)
    """
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def optimized_view_transpose(in_1):
    """
    Use unsqueeze instead of view+transpose
    in_1.view(32, -1, 1, 64).transpose(1, 2) 
    Input shape: [32, seq_len, 64]
    Target shape: [32, 1, seq_len, 64]
    
    Use unsqueeze which is a metadata-only operation
    """
    batch = 32
    
    # Unsqueeze to add dimension at position 1
    # Input: [32, seq_len, 64]
    # Output: [32, 1, seq_len, 64]
    return in_1.unsqueeze(1)


def replacement_func():
    return optimized_view_transpose