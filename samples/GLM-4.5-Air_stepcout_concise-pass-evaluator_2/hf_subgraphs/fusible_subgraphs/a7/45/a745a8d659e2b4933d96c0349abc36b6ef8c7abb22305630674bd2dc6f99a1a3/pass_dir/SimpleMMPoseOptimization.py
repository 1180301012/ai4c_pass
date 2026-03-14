import torch

# Simple pattern matching function for mmpose: linear + element-wise multiplication
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_0, None)
    tmp_3 = in_2 * tmp_1
    return (tmp_3, tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Simple optimized implementation without Triton for now
def simple_optimized_linear_mul(in_0, in_1, in_2, in_3):
    # Just do the same operations but optimized in PyTorch
    linear_result = torch.nn.functional.linear(in_3, in_0, None)
    mul_result = in_2 * in_1
    return (mul_result, linear_result)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return simple_optimized_linear_mul