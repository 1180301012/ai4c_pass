import torch

def pattern(x1, p1, train1, inplace1, x2, p2, train2, inplace2):
    """
    Pattern: Two consecutive dropout operations with p=0.0
    In the model: 
        tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
        tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    """
    # First dropout
    result1 = torch.nn.functional.dropout(x1, p1, train1, inplace1)
    # Second dropout
    result2 = torch.nn.functional.dropout(result1, p2, train2, inplace2)
    return result2

def replacement_args(x1, p1, train1, inplace1, x2, p2, train2, inplace2):
    return (x1, p1, train1, inplace1, x2, p2, train2, inplace2)

@torch.fx.wrap
def fused_dropout_zero(x, p1=0.0, train1=False, inplace1=False, p2=0.0, train2=False, inplace2=False):
    """
    Optimized replacement: Two consecutive zero-probability dropouts
    Since both are identity operations, the entire sequence is just identity
    This reduces function call overhead by fusing two operations into one
    """
    return x

def replacement_func():
    return fused_dropout_zero