import torch

# Pattern matching function - simple concat+slice pattern
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # Simplified slice pattern that should be more flexible
    slice_obj = slice(None, 672, None)
    tmp_1 = tmp_0[(slice(None, None, None), slice_obj, slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return tmp_1, tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Very simple optimized implementation using PyTorch operations
def simple_concat_optimization(in_0, in_1):
    # Simulate the slice logic without actually concatenating large tensors
    batch, c0, h, w = in_0.shape
    c1 = in_1.shape[1]
    target_channels = 672
    
    # Direct slice operation instead of concat + slice
    result_channels = min(target_channels, c0)
    
    # Slice from first input
    sliced_output = in_0[:, :result_channels, :, :]
    
    # If we need more channels from second input
    if target_channels > c0:
        from_in1 = min(target_channels - c0, c1)
        sliced_output = torch.cat([sliced_output, in_1[:, :from_in1, :, :]], dim=1)
    
    # Ensure we have exactly target_channels (pad if needed)
    if sliced_output.shape[1] < target_channels:
        padding = target_channels - sliced_output.shape[1]
        sliced_output = torch.nn.functional.pad(sliced_output, (0, 0, 0, 0, 0, padding))
    
    # Compute mean
    mean_output = sliced_output.mean((2, 3), keepdim=True)
    
    return sliced_output, mean_output

# Replacement function
def replacement_func():
    return simple_concat_optimization