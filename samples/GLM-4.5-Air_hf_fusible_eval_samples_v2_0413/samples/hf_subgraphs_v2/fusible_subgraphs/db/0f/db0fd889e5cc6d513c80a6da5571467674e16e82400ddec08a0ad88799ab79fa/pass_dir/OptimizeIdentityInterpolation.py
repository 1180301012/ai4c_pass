import torch

def pattern(input_tensor):
    # Match expand operation: expand to add batch dimension
    # This matches the expand operations in the computation
    expanded = input_tensor.expand(1, -1, -1)
    return expanded

def replacement_args(input_tensor):
    return (input_tensor,)

@torch.fx.wrap
def optimized_expand(input_tensor):
    """Optimized expand operation"""
    # If the tensor already has batch dimension 1, no need to expand
    if input_tensor.shape[0] == 1:
        return input_tensor
    # Otherwise, do the expand
    return input_tensor.expand(1, -1, -1)

def replacement_func():
    return optimized_expand