import torch


@torch.fx.wrap
def fused_add_layer_norm_wrapper(in_2, in_3, in_1, in_0, eps=1e-05):
    """
    PyTorch-based fused add + layer_norm.
    
    Computes:
    - tmp_2 = in_2 + in_3
    - tmp_4 = layer_norm(tmp_2, normalized_shape=(1024,), weight=in_1, bias=in_0, eps=eps)
    
    Returns (tmp_2, tmp_4)
    """
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, eps)
    return tmp_2, tmp_4


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match:
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_2, tmp_4)
    """
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_2, tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_2, in_3, in_1, in_0)


def replacement_func():
    return fused_add_layer_norm_wrapper