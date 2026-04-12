import torch
from pass_dir.conv_slice_kernel import conv2d_full_and_slice

def pattern(input, weight):
    conv2d = torch.conv2d(input, weight, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    return tmp_2, conv2d

def replacement_args(input, weight):
    return (input, weight, "stride1_channel1024")

def replacement_func():
    # Shared dispatch function
    def dispatch(input, weight, route):
        if route == "stride1_channel1024":
            return conv2d_full_and_slice(input, weight, (1, 1), (0, 0), (1, 1), 1, slice(None, 1024, None))
        else:
            # Fallback - return inputs unchanged (will be handled by other passes)
            return input, weight
    
    return dispatch