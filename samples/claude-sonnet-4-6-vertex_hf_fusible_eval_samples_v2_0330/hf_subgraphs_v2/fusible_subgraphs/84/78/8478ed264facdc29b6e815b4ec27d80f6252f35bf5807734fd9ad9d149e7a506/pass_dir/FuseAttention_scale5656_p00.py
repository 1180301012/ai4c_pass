import torch
import torch.nn.functional as F


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 5.656854249492381
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = torch.matmul(tmp_3, in_2)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_attention_scale5656_p00(in_0, in_1, in_2):
    # in_0: [B, H, S, D]  queries
    # in_1: [B, H, D, S2] keys (already transposed)
    # in_2: [B, H, S2, D] values
    # Convert K^T back to K for SDPA
    K = in_1.transpose(-1, -2)  # [B, H, S2, D]
    # SDPA with explicit scale overriding the default 1/sqrt(D)
    # scale param: multiplied before softmax, so scale=1/scale_factor gives Q@K^T/scale_factor
    out = F.scaled_dot_product_attention(
        in_0, K, in_2,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / 5.656854249492381,
    )
    # out: [B, H, S, D] -> permute -> [B, S, H, D] -> contiguous
    return out.permute(0, 2, 1, 3).contiguous()


def replacement_func():
    return fused_attention_scale5656_p00