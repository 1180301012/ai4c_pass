import torch

def pattern(input_tensor):
    """Pattern to match dropout operation with training=False - effectively a no-op"""
    return torch.nn.functional.dropout(input_tensor, 0.1, False, False)

def replacement_args(input_tensor):
    return (input_tensor,)

def replacement_func():
    def bypass_dropout(input_tensor):
        """Return input tensor unchanged since dropout with training=False is a no-op"""
        return input_tensor
    return bypass_dropout