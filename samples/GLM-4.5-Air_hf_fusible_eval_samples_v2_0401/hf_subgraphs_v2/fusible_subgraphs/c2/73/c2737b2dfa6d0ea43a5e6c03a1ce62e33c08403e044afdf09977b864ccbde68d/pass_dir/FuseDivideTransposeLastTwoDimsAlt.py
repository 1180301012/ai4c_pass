import torch

def pattern(in_0):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

# Using efficient PyTorch operations instead of Triton for better performance
# on simple operations like divide + transpose

@torch.fx.wrap
def fused_divide_transpose(input_tensor, scalar_value):
    """
    Efficient fused division and transpose using optimized PyTorch operations.
    Transpose is essentially a view operation (no data movement), then division.
    """
    # Create transposed tensor - this is very efficient, just changes stride info
    output_tensor = input_tensor.transpose(-1, -2)
    
    # Apply division - use in-place for better memory efficiency
    output_tensor.div_(scalar_value)
    
    return output_tensor

def replacement_func():
    # Return a closure that uses the scalar value from the pattern (2.8284271247461903)
    def kernel(in_0):
        scalar_value = 2.8284271247461903
        return fused_divide_transpose(in_0, scalar_value)
    
    return kernel