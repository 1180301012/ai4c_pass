import torch

def pattern(input_tensor):
    """
    Pattern for removing dropout with p=0.0
    Matches: dropout(input_tensor, 0.0, False, False)
    """
    return torch.nn.functional.dropout(input_tensor, 0.0, False, False)

def replacement_args(input_tensor):
    return (input_tensor,)

def identity_dropout(input_tensor):
    """
    Identity function - when dropout probability is 0.0, just return input
    This effectively eliminates the useless dropout operation
    """
    return input_tensor

def replacement_func():
    return identity_dropout