import torch
import triton
import triton.language as tl

from pass_dir.FusedAttentionWeightedSum import _fused_attn_kernel


def pattern(tmp_2, in_0):
    tmp_4 = tmp_2 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(tmp_2, in_0):
    return (tmp_2, in_0)


@torch.fx.wrap
def fused_attention_view_mul_sum_B1(tmp_2, in_0):
    B      = in_0.shape[0]
    C      = in_0.shape[1]
    H      = in_0.shape[2]
    W      = in_0.shape[3]
    out    = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)
    _fused_attn_kernel[(B * C,)](in_0, tmp_2, out, B, C, H * W)
    return out


def replacement_func():
    return fused_attention_view_mul_sum_B1