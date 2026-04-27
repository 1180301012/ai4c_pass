import torch
from pass_dir._shared import _dispatch


def pattern(in_0, in_4, in_5, in_3, in_2, in_1, arange0, arange1):
    """
    arange0 / arange1 match _tensor_constant0 / _tensor_constant1 (get_attr nodes).
    """
    # ── Embedding + LayerNorm (DIM=64, eps=1e-12) ────────────────
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (64,), in_2, in_1, 1e-12)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)

    # ── Position bucket (no torch.arange here) ───────────────────
    tmp_11 = arange0[(slice(None, None, None), None)]
    tmp_13 = arange1[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_32 = tmp_19 + tmp_31

    return tmp_9, tmp_32


def replacement_args(in_0, in_4, in_5, in_3, in_2, in_1, arange0, arange1):
    return (in_0, in_4, in_5, in_3, in_2, in_1, arange0, "route_64_1e12")


def replacement_func():
    return _dispatch