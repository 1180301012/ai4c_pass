import torch

def pattern(tmp_10):
    tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0), 'constant', None)
    return tmp_11

def replacement_args(tmp_10):
    return (tmp_10,)

@torch.fx.wrap
def remove_pad(input_tensor):
    # Pad with all zeros is a no-op, just return input directly
    return input_tensor

def replacement_func():
    return remove_pad