import torch

def pattern(x, p, training):
    return torch.nn.functional.dropout(x, p=p, training=training)

def replacement_args(x, p, training):
    return (x, p, training)

def noop_dropout(x, p, training):
    return x

def replacement_func():
    return noop_dropout