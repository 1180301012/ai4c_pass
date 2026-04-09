import torch

def pattern(tmp_1):
    # Pattern matches dropout followed by flatten
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(tmp_1):
    return (tmp_1,)

@torch.fx.wrap
def dropout_flatten(tmp_1):
    """Fuse dropout (with training=False) and flatten operations"""
    # When training=False, dropout is just identity operation
    # Skip the dropout and just flatten directly - this eliminates one operation
    return torch.flatten(tmp_1, 1)

def replacement_func():
    return dropout_flatten