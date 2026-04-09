import torch

def pattern(tmp_8):
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9

def replacement_args(tmp_8):
    return (tmp_8,)

@torch.fx.wrap
def remove_dropout(input_tensor):
    # Dropout with p=0.0 is a no-op, just return input directly
    return input_tensor

def replacement_func():
    return remove_dropout