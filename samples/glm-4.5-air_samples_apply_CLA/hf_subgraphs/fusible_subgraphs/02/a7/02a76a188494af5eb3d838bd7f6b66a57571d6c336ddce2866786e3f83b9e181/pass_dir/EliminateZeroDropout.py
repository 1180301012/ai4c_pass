import torch

def pattern(embedding_out):
    # Match dropout operation with p=0.0
    dropout_out = torch.nn.functional.dropout(embedding_out, 0.0, False, False)
    return dropout_out

def replacement_args(embedding_out):
    return (embedding_out,)

def identity_function(x):
    """Simple identity function that just returns input"""
    return x

def replacement_func():
    return identity_function