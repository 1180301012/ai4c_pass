import torch
from pass_dir.fused_ops_kernel import dispatch_kernel

def pattern(in_2, in_3, conv2d, tmp_3):
    """Pattern for cat + reshape + transpose sequence"""
    tmp_4 = tmp_3.reshape(1, 8, -1, -1)
    tmp_5 = tmp_4.transpose(-1, -2)
    return tmp_5

def replacement_args(in_2, in_3, conv2d, tmp_3):
    # Extract shape info for the wrapper
    # tmp_3 has shape [1, C2+C3+Cout, H, W]
    # After reshape [1, 8, K, N] where K = (C2+C3+Cout)/8 and N = H*W
    B, C, H, W = tmp_3.shape
    K = C // 8
    N = H * W
    return (in_2, in_3, conv2d, K, N, "cat_reshape_transpose")

def replacement_func():
    return dispatch_kernel