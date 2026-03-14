import torch

def pattern(in_4, in_5):
    """
    Pattern to match: add + spatial mean + dropout(p=0) + dropout(p=0)
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7

def replacement_args(in_4, in_5):
    return (in_4, in_5)

@torch.fx.wrap
def fused_add_spatial_mean(in_4, in_5):
    """
    Use PyTorch's efficient primitives - fuse add+mean in single expression
    Dropout with p=0 is no-op, so we skip it
    """
    return (in_4 + in_5).mean(dim=(2, 3), keepdim=False)

def replacement_func():
    return fused_add_spatial_mean