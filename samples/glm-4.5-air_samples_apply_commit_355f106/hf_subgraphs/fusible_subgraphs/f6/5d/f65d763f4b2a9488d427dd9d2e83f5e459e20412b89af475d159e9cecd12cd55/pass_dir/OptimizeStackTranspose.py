import torch
import triton
import triton.language as tl

# Pattern matching for stack-transpose optimization
def pattern(a, b):
    stacked = torch.stack((a, b), dim=-1)
    transposed = stacked.transpose(-1, -2)
    return transposed

# Argument extraction
def replacement_args(a, b):
    return (a, b)

# Optimized stack-transpose operation using direct tensor construction
def optimized_stack_transpose(a, b):
    """
    Directly constructs [H, 2, W] from two [H, W] tensors,
    avoiding intermediate [H, W, 2] creation and transpose
    """
    # The original computation:
    # stacked = torch.stack((a, b), dim=-1)    # [H, W, 2] 
    # transposed = stacked.transpose(-1, -2)    # [H, 2, W]
    
    # Optimized version: directly create the [H, 2, W] result
    # This avoids the temporary [H, W, 2] allocation and transpose operation
    result = torch.stack((a, b), dim=1)  # Creates [H, 2, W] directly
    return result

def replacement_func():
    return optimized_stack_transpose