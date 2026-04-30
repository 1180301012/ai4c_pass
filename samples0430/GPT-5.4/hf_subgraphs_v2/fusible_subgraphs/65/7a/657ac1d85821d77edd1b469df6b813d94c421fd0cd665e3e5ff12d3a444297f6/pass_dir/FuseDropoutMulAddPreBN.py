import torch
from pass_dir.shared_tail_fusion import fused_tail_dispatch


def pattern(conv_out, layer_scale, residual):
    tmp_8 = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    tmp_9 = tmp_8 * layer_scale
    tmp_10 = residual + tmp_9
    return tmp_10


def replacement_args(conv_out, layer_scale, residual):
    layer_scale_1d = layer_scale.reshape(-1)
    zeros = layer_scale_1d * 0.0
    return (conv_out, layer_scale_1d, zeros, zeros, residual, "pre_bn")


def replacement_func():
    return fused_tail_dispatch