import torch

def pattern(in_0):
    """
    Simple pattern to test ReLU operation matching
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace = True)
    return (tmp_0,)

def replacement_args(in_0):
    """
    Extract input tensor for the replacement
    """
    return (in_0,)

def replacement_func():
    """
    Returns reference to a simple replacement function
    """
    def simple_relu(x):
        return torch.nn.functional.relu(x, inplace=True)
    return simple_relu