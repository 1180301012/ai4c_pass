import torch

# Pattern matching for dropout elimination
def pattern(conv_out):
    """
    Match dropout operation with p=0.0, which is effectively a no-op
    """
    # Dropout with p=0.0 is equivalent to identity operation
    dropout_out = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    return dropout_out

# Argument extraction function  
def replacement_args(conv_out):
    return (conv_out,)

# Replacement function - just return the input since dropout with p=0.0 is identity
def replacement_func():
    def identity_dropout(conv_out):
        return conv_out
    return identity_dropout