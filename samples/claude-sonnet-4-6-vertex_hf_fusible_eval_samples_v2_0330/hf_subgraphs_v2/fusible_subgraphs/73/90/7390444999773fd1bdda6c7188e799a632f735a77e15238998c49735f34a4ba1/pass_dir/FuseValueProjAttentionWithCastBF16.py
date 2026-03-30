import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl
import math
from FuseValueProjAttentionNoCast import (
    flash_attn_kernel, bhsd_to_bshd_kernel, _to_gpu
)


@torch.fx.wrap
def fused_value_proj_attention_bf16(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Fused value projection + attention with absorbed bfloat16 cast.
    in_0: bias   [H*D]  in_1: weight [H*D, H*D]
    in_2: mask   [B,1,S,S]
    in_3: hidden [B,S,H*D]   in_4: key  [B,H,S,D]   in_5: query [B,H,S,D]
    """
    device    = in_3.device
    out_dtype = torch.bfloat16

    w = _to_gpu(in_1, device, out_dtype)
    b = _to_gpu(in_0, device, out_dtype)

    B = in_3.shape[0];  S = in_3.shape[1]
    H = in_4.shape[1];  D = in_4.shape[3]

    # Value projection (absorb the cast into @ operator)
    in3_bf16 = in_3.to(out_dtype) if in_3.dtype != out_dtype else in_3
    lin_2d   = in3_bf16.reshape(B * S, H * D) @ w.t() + b
    value    = lin_2d.view(B, S, H, D).transpose(1, 2).contiguous()

    # Flash Attention
    attn_out = torch.empty_like(value)
    sm_scale = 1.0 / math.sqrt(D)
    Q = in_5.contiguous();  K = in_4.contiguous()

    flash_attn_kernel[(triton.cdiv(S, 64), B * H)](
        Q, K, value, in_2, attn_out,
        B, H, S, D=D, sm_scale=sm_scale,
        stride_maskb=in_2.stride(0),
        stride_maskm=in_2.stride(2),
        stride_maskn=in_2.stride(3),
        B_H=B * H,
    )

    # Fused transpose+reshape
    out = torch.empty(B, S, H * D, dtype=attn_out.dtype, device=device)
    bhsd_to_bshd_kernel[(B, H, triton.cdiv(S, 32))](
        attn_out, out, B, H, S, D=D,
    )
    return (out,)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3  = linear.view(1, -1, 2, 64)
    tmp_4  = tmp_3.transpose(1, 2)
    to     = tmp_4.to(torch.bfloat16)
    sdpa   = torch.nn.functional.scaled_dot_product_attention(
                 in_5, in_4, to, attn_mask=in_2,
                 dropout_p=0.0, is_causal=False)
    tmp_6  = sdpa.transpose(1, 2)
    tmp_7  = tmp_6.reshape(1, -1, 128)
    return (tmp_7,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_value_proj_attention_bf16