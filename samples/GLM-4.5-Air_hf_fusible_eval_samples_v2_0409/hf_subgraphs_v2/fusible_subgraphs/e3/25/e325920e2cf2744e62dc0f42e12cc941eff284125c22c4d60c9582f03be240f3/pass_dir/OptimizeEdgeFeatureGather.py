import torch

def pattern(in_0, in_1):
    """Matches the pattern exactly as in the model"""
    tmp_0 = in_0[1]
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(-2, tmp_1)
    return (tmp_0, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    """Return a function that performs the same computation as the original pattern"""
    def original_computation(in_0, in_1):
        tmp_0 = in_0[1]
        tmp_1 = in_0[0]
        tmp_2 = in_1.index_select(-2, tmp_1)
        return (tmp_0, tmp_2)
    return original_computation