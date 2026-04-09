import torch

def pattern(tmp_9):
    # Match: view -> pad -> view sequence
    tmp_10 = tmp_9.view(1, 16, 16, 16)
    tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
    tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
    return tmp_12

def replacement_args(tmp_9):
    return (tmp_9,)

@torch.fx.wrap
def optimize_view_pad_view_sequence(input_tensor):
    # Skip the intermediate pad (no-op) and view operations
    # Go directly from [1, 256, 16] to [1, 8, 2, 8, 2, 16]
    output = input_tensor.reshape(1, 8, 2, 8, 2, 16)
    return output

def replacement_func():
    return optimize_view_pad_view_sequence