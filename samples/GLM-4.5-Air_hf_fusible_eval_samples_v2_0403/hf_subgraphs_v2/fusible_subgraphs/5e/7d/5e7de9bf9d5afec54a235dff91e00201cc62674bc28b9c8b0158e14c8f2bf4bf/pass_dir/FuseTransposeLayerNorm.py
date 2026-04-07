import torch

# Pattern matching function for transpose + layer norm fusion
def pattern(tmp_8, in_1, in_0):
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    return tmp_10

# Argument extraction function
def replacement_args(tmp_8, in_1, in_0):
    return (tmp_8, in_1, in_0)

# Replacement function
def replacement_func():
    def simple_fused_transpose_layernorm(tmp_8, in_1, in_0):
        tmp_9 = tmp_8.transpose(1, 2)
        return torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    return simple_fused_transpose_layernorm