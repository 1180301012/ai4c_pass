import torch

def pattern(x, p=0.0, training=False):
    return torch.nn.functional.dropout(x, p=p, training=training)

def replacement_args(x, p=0.0, training=False):
    return (x, p, training)

def replacement_func():
    def identity_dropout(x, p=0.0, training=False):
        # Dropout with p=0.0 is just identity operation
        return x
    
    return identity_dropout