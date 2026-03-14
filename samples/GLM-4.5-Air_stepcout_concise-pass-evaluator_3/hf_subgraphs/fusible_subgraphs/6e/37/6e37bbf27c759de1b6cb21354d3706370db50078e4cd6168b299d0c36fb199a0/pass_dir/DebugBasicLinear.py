import torch

def pattern(in_0, in_1, in_2, in_3):
    # Match the linear operation exactly
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_2, tmp_0, None)
    
    # Return the 3 outputs as in original
    tmp_4 = in_1.unsqueeze(1)
    tmp_5 = in_3.unsqueeze(1)
    
    return (tmp_4, tmp_5, tmp_1)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Simple replacement using basic tensor operations
def optimized_linear(x, weight):
    # Implement matrix multiplication manually using basic tensor operations
    # This is equivalent to torch.nn.functional.linear(x, weight, None)
    return x @ weight.t()

def replacement_func():
    def wrapper(in_0, in_1, in_2, in_3):
        # Replace linear operation with optimized version
        linear_result = optimized_linear(in_2, in_0)
        
        # Keep the unsqueeze operations as in original
        tmp_4 = in_1.unsqueeze(1)
        tmp_5 = in_3.unsqueeze(1)
        
        return (tmp_4, tmp_5, linear_result)
    return wrapper