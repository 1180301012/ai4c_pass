import torch

def pattern(x, p, training, inplace):
    # Dropout with rate 0.0 is effectively a no-op
    # The pattern matches exactly how dropout is called in the original graph
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x, p, training, inplace):
    # We need all arguments to match the exact call pattern
    return (x, 0.0, False, False)

def replacement_func():
    # Dropout with rate 0.0 just returns the input unchanged
    # Must accept all 4 arguments to match the original call signature
    def eliminate_dropout(x, p, training, inplace):
        return x
    return eliminate_dropout