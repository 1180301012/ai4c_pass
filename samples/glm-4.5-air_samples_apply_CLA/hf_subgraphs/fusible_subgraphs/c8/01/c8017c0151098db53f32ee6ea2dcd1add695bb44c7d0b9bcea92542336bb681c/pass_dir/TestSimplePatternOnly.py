import torch

# Test pattern: Try to match the exact variable assignments from the model
def pattern(x, conv_bias, conv_weight):
    # Replicate the exact variable assignment pattern from the model
    tmp_0 = conv_bias
    tmp_1 = conv_weight
    tmp_2 = torch.conv2d(x, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.nn.functional.hardsigmoid(tmp_2, False)
    tmp_2 = None
    return tmp_3

# Extract arguments
def replacement_args(x, conv_bias, conv_weight):
    return (x, conv_bias, conv_weight)

# Simple replacement
def replacement_func():
    return pattern