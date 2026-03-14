import torch
import triton
import triton.language as tl


def fused_conv2d_maxpool_2x2_stride_3x3_padding(x, weight):
    """
    Fused Conv2d (stride=2, padding=3) + MaxPool2d (kernel=3, stride=2, padding=1)
    
    Input: (N, C, H, W)
    Weight: (CO, C, KH, KW) - typically KH=7, KW=7 for stem conv
    Conv: stride=(2,2), padding=(3,3), dilation=(1,1), groups=1
    Pool: kernel=3, stride=2, padding=1
    """
    N, C, H, W = x.shape
    CO, C_in, KH, KW = weight.shape
    
    # Conv2d
    conv_out = torch.nn.functional.conv2d(x, weight, stride=(2, 2), padding=(3, 3), dilation=(1, 1), groups=1)
    
    # MaxPool2d
    pool_out = torch.nn.functional.max_pool2d(conv_out, kernel_size=3, stride=2, padding=1, ceil_mode=False, return_indices=False)
    
    return pool_out


@torch.fx.wrap
def fused_conv2d_maxpool_wrapper_2x2(in_0, in_1):
    """
    Wrapper for fused Conv2d (stride=2, padding=3) + MaxPool2d operation.
    in_0: weight tensor (CO, C, KH, KW)
    in_1: input tensor (N, C, H, W)
    """
    return fused_conv2d_maxpool_2x2_stride_3x3_padding(in_1, in_0)


def pattern(in_0, in_1):
    """
    Pattern: Conv2d (stride=2, padding=3) + MaxPool2d (kernel=3, stride=2, padding=1)
    
    For resnetv2_101.a1h_in1k_start0_end2_0:
    - Conv2d: stride=(2,2), padding=(3,3), dilation=(1,1), groups=1
    - MaxPool2d: kernel_size=3, stride=2, padding=1
    """
    tmp_0 = in_0  # weight
    tmp_1 = in_1  # input
    tmp_2 = torch.conv2d(tmp_1, tmp_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.nn.functional.max_pool2d(tmp_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    tmp_2 = None
    return tmp_3


def replacement_args(in_0, in_1):
    """
    Extract arguments for replacement function.
    in_0: weight tensor (CO, C, KH, KW)
    in_1: input tensor (N, C, H, W)
    """
    return (in_0, in_1)


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_conv2d_maxpool_wrapper_2x2