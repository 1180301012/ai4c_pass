import torch
import torch.nn.functional as F


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 6.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    tmp_4 = torch.matmul(tmp_3, in_2)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_attention_scale6_p01(in_0, in_1, in_2):
    # in_0: [B, H, S, D]  queries
    # in_1: [B, H, D, S2] keys (already transposed)
    # in_2: [B, H, S2, D] values
    # dropout p=0.1 with training=False is a no-op, so use p=0.0
    K = in_1.transpose(-1, -2)  # [B, H, S2, D]
    out = F.scaled_dot_product_attention(
        in_0, K, in_2,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / 6.0,
    )
    # out: [B, H, S, D] -> permute -> [B, S, H, D] -> contiguous
    return out.permute(0, 2, 1, 3).contiguous()


def replacement_func():
    return fused_attention_scale6_p01