import torch
from torch import device

from pass_dir.causal_mask_common import causal_attention_mask_dispatch_routed


NEG_INF = -3.4028234663852886e+38


def pattern(in_0: torch.Tensor):
    tmp_1 = torch.arange(0, 10, device=device(type='cuda', index=0))
    tmp_2 = torch.full((10, 10), fill_value=NEG_INF, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(10, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_7 = tmp_3 * tmp_6
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))]
    tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))]
    tmp_17 = tmp_16.masked_fill(tmp_15, NEG_INF)
    tmp_18 = torch.slice_scatter(tmp_10, tmp_17, dim=3, start=0, end=10, step=1)
    tmp_19 = tmp_18.__eq__(NEG_INF)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_18.mul(tmp_21)
    return tmp_22


def replacement_args(in_0: torch.Tensor):
    return (in_0, 'L10')


def replacement_func():
    return causal_attention_mask_dispatch_routed