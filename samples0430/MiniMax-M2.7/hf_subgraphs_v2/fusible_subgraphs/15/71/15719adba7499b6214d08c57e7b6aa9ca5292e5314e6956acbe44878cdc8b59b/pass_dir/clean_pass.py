import torch
import triton
import triton.language as tl
from pass_dir.helper_ops import conv2d_fwd, layer_norm_fwd, dropout_fwd, pad_fwd

# Pattern to match the full computation
def pattern(in_0, in_1, in_2, in_3, in_4):
    # Conv2D with bias
    conv2d = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    tmp_10 = tmp_9.view(1, 16, 16, 16)
    tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
    tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
    return (tmp_9, tmp_13)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# Module-level function for replacement
def pass_through_impl(in_0, in_1, in_2, in_3, in_4):
    conv2d = conv2d_fwd(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = layer_norm_fwd(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = dropout_fwd(tmp_8, 0.0, False, False)
    tmp_10 = tmp_9.view(1, 16, 16, 16)
    tmp_11 = pad_fwd(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
    tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
    return (tmp_9, tmp_13)


def replacement_func():
    return pass_through_impl