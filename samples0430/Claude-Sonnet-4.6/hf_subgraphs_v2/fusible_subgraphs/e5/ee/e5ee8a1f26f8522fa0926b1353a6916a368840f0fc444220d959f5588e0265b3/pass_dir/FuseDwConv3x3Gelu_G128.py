import torch
import triton
import triton.language as tl
from pass_dir.shared_fused_ops import universal_fused_op


def pattern(in_0, in_1, in_2):
    # in_0=bias[128], in_1=weight[128,1,3,3], in_2=input[N,128,H,W]
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 128)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    # x=in_2 (input), ctx1=in_1 (weight), ctx2=in_0 (bias)
    return (in_2, in_1, in_0, 'dw_conv3x3_gelu_128')


def replacement_func():
    return universal_fused_op