import torch

def pattern(in_0, in_1, in_2, in_3):
    """Matches the pattern with useless torch.rand([]) operation"""
    tmp_2 = in_2 + in_3
    tmp_3 = torch.rand([])
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_2, tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    @torch.fx.wrap
    def remove_useless_random(bias, weight, x1, x2):
        # Simply perform the computation without the useless random operation
        tmp_2 = x1 + x2
        tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), weight, bias, 1e-05)
        return tmp_2, tmp_4
    
    return remove_useless_random