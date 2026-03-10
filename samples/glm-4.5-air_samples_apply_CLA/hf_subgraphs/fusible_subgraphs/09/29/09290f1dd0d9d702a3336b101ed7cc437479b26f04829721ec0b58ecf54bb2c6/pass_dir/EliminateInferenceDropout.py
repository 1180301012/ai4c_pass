import torch

def pattern(input_tensor):
    # Dropout with 20% rate during training
    # During inference, dropout becomes identity operation
    output = torch.nn.functional.dropout(input_tensor, 0.2, False, False)
    return output

def replacement_args(input_tensor):
    return (input_tensor,)

def replacement_func():
    # During inference, dropout is a no-op - just return the input
    def identity_dropout(input_tensor):
        return input_tensor
    
    return identity_dropout