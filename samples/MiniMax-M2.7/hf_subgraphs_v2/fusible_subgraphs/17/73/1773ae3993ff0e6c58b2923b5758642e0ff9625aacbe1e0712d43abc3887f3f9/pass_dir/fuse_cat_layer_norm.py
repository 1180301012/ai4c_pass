import torch


# ============================================================================
# Cat Pattern: torch.cat((in_2, in_5, in_3), dim=2)
# ============================================================================
def pattern(in_2, in_5, in_3):
    """Pattern matcher for torch.cat along dim=2"""
    tmp_2 = torch.cat((in_2, in_5, in_3), dim=2)
    return tmp_2


def replacement_args(in_2, in_5, in_3):
    """Extract arguments for cat replacement"""
    return (in_2, in_5, in_3)


# ============================================================================
# Replacement Function (uses PyTorch's built-in cat)
# ============================================================================
def replacement_func():
    """Returns a function that uses PyTorch's built-in cat"""
    def _cat_pytorch(in_2, in_5, in_3):
        return torch.cat((in_2, in_5, in_3), dim=2)
    return _cat_pytorch