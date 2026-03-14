import torch


def pattern(in_4, in_5):
    """
    Pattern: tmp_4 = in_5 + in_4; tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    
    This pattern just returns the original computation to test baseline.
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    return (in_4, in_5)


def replacement_func():
    # Just use original PyTorch code - it's already optimized
    def original_add_mean(in_4, in_5):
        tmp_4 = in_5 + in_4
        tmp_5 = tmp_4.mean((2, 3), keepdim=False)
        return tmp_5
    return original_add_mean