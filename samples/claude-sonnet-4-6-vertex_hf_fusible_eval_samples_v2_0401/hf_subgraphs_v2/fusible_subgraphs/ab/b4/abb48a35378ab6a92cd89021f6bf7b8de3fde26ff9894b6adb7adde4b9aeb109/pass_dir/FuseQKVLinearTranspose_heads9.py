import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    tmp_5 = unbind[0]
    tmp_6 = unbind[1]
    tmp_7 = unbind[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def _compute_qkv_9heads(in_0, in_1):
    weight = in_0.to(device=in_1.device, dtype=in_1.dtype)
    B, S, C = in_1.shape[0], in_1.shape[1], in_1.shape[2]
    out = in_1.reshape(B * S, C) @ weight.t()
    out = out.reshape(B, S, 3, 9, 48).permute(2, 0, 3, 1, 4)
    parts = out.unbind(0)
    Q   = parts[0]
    K_T = parts[1].transpose(-2, -1)
    V   = parts[2]
    return Q, K_T, V


def qkv_linear_9heads(in_0, in_1):
    result = _compute_qkv_9heads(in_0, in_1)
    Q   = result[0]
    K_T = result[1]
    V   = result[2]
    return Q, K_T, V


def replacement_func():
    return qkv_linear_9heads