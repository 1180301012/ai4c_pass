import torch
import triton
import triton.language as tl

# Pattern matching function - matches select channel 0 + unsqueeze sequence
def pattern(input_tensor):
    # Matches: select index 0 on dim 1, then unsqueeze at dim 1
    tmp_1 = input_tensor[slice(None, None, None), 0]
    tmp_2 = torch.unsqueeze(tmp_1, 1)
    return tmp_2

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized direct slicing approach - much more efficient for simple operations
@torch.fx.wrap
def select_channel_optimized(input_tensor):
    """
    Optimized version that directly slices channel 0 with dimension preservation
    Instead of: tmp_1 = input[:, 0], then tmp_2 = unsqueeze(tmp_1, 1)
    We directly do: input[:, 0:1, :, :] to get the same result
    """
    # Direct slicing to get [batch, 1, height, width] in one operation
    # This eliminates the intermediate tensor and separate unsqueeze operation
    return input_tensor[:, 0:1, :, :]

# Replacement function returns the optimized function
def replacement_func():
    return select_channel_optimized