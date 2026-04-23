import torch


def exact_mha_no_dropouts(out_proj_bias, out_proj_weight, in_proj_bias, in_proj_weight, obj_feat):
    out = torch.nn.functional.multi_head_attention_forward(
        obj_feat,
        obj_feat,
        obj_feat,
        512,
        8,
        in_proj_weight,
        in_proj_bias,
        None,
        None,
        False,
        0.0,
        out_proj_weight,
        out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    )[0]
    return (out,)