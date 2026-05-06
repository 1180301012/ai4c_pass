"""
SE One-Scale Pass: fuses conv2d(1x1) + (+1.0 / 2.0) + clamp_(0,1) + mul SE-block.
Matches: bfloat16 S-ViPNAS, float16 S-ViPNAS, float32 S-ViPNAS (mobilenetv3-start0)
Scale constants: add=1.0, div=2.0  =>  scale = (conv_out + 1.0) / 2.0
"""
import torch
import triton
import triton.language as tl

from pass_dir.se_kernel import fused_se_forward


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern mirrors model.py exactly:
      conv2d(in_3, in_1, in_0, stride=1, pad=0, dil=1, groups=1)
      -> +1.0 -> /2.0 -> clamp_(0,1) -> in_2 * clamp
    in_0 = bias  [Cout]
    in_1 = weight [Cout, Cin, 1, 1]
    in_2 = se_mul_input [B, Cout, H, W]
    in_3 = conv_input [B, Cin, 1, 1]
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d + 1.0
    tmp_4  = tmp_3 / 2.0
    tmp_5  = tmp_4.clamp_(0.0, 1.0)
    tmp_6  = in_2 * tmp_5
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3):
    # Order matches fused_se_forward(bias, weight, feature_map, in2, eps_scale)
    return (in_0, in_1, in_3, in_2, 1.0)


@torch.fx.wrap
def _fused_se_wrapper_one(in_0, in_1, in_3, in_2, eps_scale):
    return fused_se_forward(in_0, in_1, in_3, in_2, eps_scale)


def replacement_func():
    return _fused_se_wrapper_one