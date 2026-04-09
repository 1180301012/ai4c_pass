import torch
import triton
import triton.language as tl

def pattern(in_0, in_4, in_3):
    conv2d = torch.conv2d(in_0, in_4, in_3, (4, 4), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

def replacement_args(in_0, in_4, in_3):
    return (in_0, in_4, in_3)

@torch.fx.wrap
def conv2d_flatten_transpose_fused_96(in_0, in_4, in_3):
    # Use optimized PyTorch operations for 96-channel case with stride 4x4
    
    # Use optimized conv2d followed by flatten and transpose
    conv2d = torch.nn.functional.conv2d(in_0, in_4, in_3, stride=(4, 4), padding=(0, 0), dilation=(1, 1), groups=1)
    
    # Flatten and transpose
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    
    return tmp_7

def replacement_func():
    return conv2d_flatten_transpose_fused_96