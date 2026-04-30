import torch

def pattern(x, normalized_shape, weight, bias, eps):
    """
    Simple layer_norm pattern - just match the layer_norm operation
    """
    ln_out = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    return ln_out

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

def replacement_func():
    def fallback_layer_norm(x, normalized_shape, weight, bias, eps):
        return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    return fallback_layer_norm