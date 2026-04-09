import torch

def pattern(tmp_7, in_2, in_1):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9

def replacement_args(tmp_7, in_2, in_1):
    return (tmp_7, in_2, in_1)

@torch.fx.wrap
def remove_layer_norm_dropout(input_tensor, weight, bias):
    # LayerNorm followed by dropout(p=0.0) is equivalent to just LayerNorm
    # Since dropout with p=0.0 is a no-op
    output = torch.nn.functional.layer_norm(input_tensor, weight.shape, weight, bias, 1e-05)
    return output

def replacement_func():
    return remove_layer_norm_dropout