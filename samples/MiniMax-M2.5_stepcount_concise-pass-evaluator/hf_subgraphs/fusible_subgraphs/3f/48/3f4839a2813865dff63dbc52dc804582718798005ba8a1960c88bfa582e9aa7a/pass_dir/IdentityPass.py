import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Identity pattern - just to test matching.
    Returns the same computation as the model.
    """
    # Match exactly
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    tmp_3 = tmp_2 - in_2
    tmp_4 = tmp_0.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 * tmp_3
    tmp_7 = in_2 + tmp_6
    tmp_8 = tmp_1.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    return tmp_7, tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    """Identity replacement - just use PyTorch ops directly."""
    
    @torch.fx.wrap
    def identity_replacement(scale1, scale2, in_2):
        """Exact same computation as model - no optimization."""
        tmp_0 = scale1
        tmp_1 = scale2
        tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
        tmp_3 = tmp_2 - in_2
        tmp_4 = tmp_0.unsqueeze(-1)
        tmp_5 = tmp_4.unsqueeze(-1)
        tmp_6 = tmp_5 * tmp_3
        tmp_7 = in_2 + tmp_6
        tmp_8 = tmp_1.unsqueeze(-1)
        tmp_9 = tmp_8.unsqueeze(-1)
        return tmp_7, tmp_9
    
    return identity_replacement