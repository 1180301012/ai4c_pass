import torch
from pass_dir.shared_sdpa_fused_out import replacement_func


def pattern(q, k, v, mask):
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    t = a.transpose(1, 2)
    y = t.reshape(4, 512, 256)
    return y


def replacement_args(q, k, v, mask):
    return (q, k, v, mask, 'sdpa_fused_out')