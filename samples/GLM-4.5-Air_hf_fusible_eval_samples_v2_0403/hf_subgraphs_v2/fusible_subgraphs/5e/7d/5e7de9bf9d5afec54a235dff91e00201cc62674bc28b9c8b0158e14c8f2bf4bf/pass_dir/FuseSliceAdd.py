import torch

# Pattern matching function for slicing + addition fusion
def pattern(tmp_4, in_3):
    tmp_5 = torch.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
    tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))]
    tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
    tmp_8 = tmp_6 + tmp_7
    return tmp_8

# Argument extraction function
def replacement_args(tmp_4, in_3):
    return (tmp_4, in_3)

# Replacement function
def replacement_func():
    def simple_fused_slice_add(tmp_4, in_3):
        tmp_5 = torch.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
        tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))]
        tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
        return tmp_6 + tmp_7
    return simple_fused_slice_add