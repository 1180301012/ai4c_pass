import torch
from pass_dir.fused_kernel import fused_conv_ln_relu_dispatch

# Pattern for normalized_shape = (38, 1, 1)
def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    ln_out = torch.nn.functional.layer_norm(conv2d, (38, 1, 1), in_3, in_2, 1e-05)
    relu_out = torch.nn.functional.relu(ln_out, inplace=True)
    return (relu_out,)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # in_0 = conv_bias, in_1 = conv_weight, in_2 = ln_bias, in_3 = ln_weight, in_4 = input
    return (in_0, in_1, in_2, in_3, in_4, "route_C38")

def replacement_func():
    return fused_conv_ln_relu_dispatch