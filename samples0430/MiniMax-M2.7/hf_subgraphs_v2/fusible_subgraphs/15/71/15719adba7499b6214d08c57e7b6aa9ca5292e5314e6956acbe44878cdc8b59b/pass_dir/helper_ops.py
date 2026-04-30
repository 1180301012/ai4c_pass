import torch

def conv2d_fwd(in_0, in_4, in_3, stride, padding, dilation, groups):
    return torch.conv2d(in_0, in_4, in_3, stride, padding, dilation, groups)

def layer_norm_fwd(tmp_7, normalized_shape, in_2, in_1, eps):
    return torch.nn.functional.layer_norm(tmp_7, normalized_shape, in_2, in_1, eps)

def dropout_fwd(tmp_8, p, training, inplace):
    return torch.nn.functional.dropout(tmp_8, p, training, inplace)

def pad_fwd(tmp_10, pad, mode, value):
    return torch.nn.functional.pad(tmp_10, pad, mode, value)