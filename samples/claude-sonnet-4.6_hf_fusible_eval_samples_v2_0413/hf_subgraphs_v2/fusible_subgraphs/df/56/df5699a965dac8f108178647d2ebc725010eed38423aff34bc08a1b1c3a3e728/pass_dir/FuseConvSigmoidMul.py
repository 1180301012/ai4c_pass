"""FuseConvSigmoidMul: fuses sigmoid(conv_out) * in5 -> single output [B,40,32,24]"""
import torch
import sys, os
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from shared_kernels import fused_sigmoid_mul_impl

def pattern(in5, conv_out):
    tmp_3 = torch.sigmoid(conv_out)
    tmp_4 = in5 * tmp_3
    return tmp_4

def replacement_args(in5, conv_out):
    return (in5, conv_out)

def replacement_func():
    return fused_sigmoid_mul_impl