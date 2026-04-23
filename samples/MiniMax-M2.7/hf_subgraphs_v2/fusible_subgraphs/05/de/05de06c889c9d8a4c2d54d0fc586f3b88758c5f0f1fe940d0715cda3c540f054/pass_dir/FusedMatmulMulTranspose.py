import torch

# Pattern matching function - matches the exact computation pattern
def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    tmp_2 = tmp_1.T
    return tmp_1, tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Kernel wrapper using torch.fx.wrap - using native PyTorch operations
@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2):
    # Fused matmul + multiply + transpose
    # in_0: scalar, in_1: [512, 1], in_2: [2, 512]
    result = torch.matmul(in_2, in_1) * in_0
    return result, result.T


# Replacement function
def replacement_func():
    return kernel_wrapper