import torch
from pass_dir.fused_ops_kernel import dispatch_kernel

def pattern(in_6, in_4, tmp_5, tmp_6, tmp_7, tmp_8, tmp_9, tmp_10, tmp_11, scalar):
    """Pattern for multiply + pad + scale + add + transpose + reshape sequence"""
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scalar * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, -1, -1)  # Match any reshape dimensions
    return tmp_9, tmp_10, tmp_11

def replacement_args(in_6, in_4, tmp_5, scalar):
    # Extract shape info
    # tmp_5 has shape [1, 8, N, K] after transpose
    # in_4 has shape [1, 8, N+1, K] (one more in N dim due to padding)
    # scalar is the scale factor
    B, H, N, K = tmp_5.shape
    return (tmp_5, in_6, in_4, scalar, "mult_pad_scale_add")

def replacement_func():
    return dispatch_kernel