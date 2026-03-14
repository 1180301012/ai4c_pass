import torch

def pattern(in_0):
    tmp_0 = in_0 * 1.0
    return tmp_0

@torch.fx.wrap
def test_functionality(in_0):
    """Test working pass"""
    return in_0

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return test_functionality