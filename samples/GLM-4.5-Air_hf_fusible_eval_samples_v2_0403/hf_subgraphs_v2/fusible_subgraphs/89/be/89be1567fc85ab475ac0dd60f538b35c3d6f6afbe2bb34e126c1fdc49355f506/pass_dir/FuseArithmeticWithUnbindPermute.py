import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matches the fused computation:
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0  
    unbind = torch.unbind(tmp_2, dim=2)
    tmp_4 = unbind[0]
    tmp_5 = unbind[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)
    """
    tmp_0 = in_0
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + tmp_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    def simple_replacement(in_0, in_1, in_2):
        """
        Simple replacement that just repeats the original computation
        to test pattern matching first
        """
        tmp_0 = in_0
        tmp_1 = in_2 * in_1
        tmp_2 = tmp_1 + tmp_0
        tmp_3 = torch.unbind(tmp_2, dim=2)
        tmp_4 = tmp_3[0]
        tmp_5 = tmp_3[1]
        tmp_6 = tmp_5.permute(0, 2, 1)
        return (tmp_6, tmp_4)
    
    return simple_replacement