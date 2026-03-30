import torch
import triton
import triton.language as tl


def pattern():
    tmp_10 = torch.arange(7, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(7, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
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
    tmp_19 += tmp_31
    tmp_32 = tmp_19
    return tmp_32


def replacement_args():
    return ()


# Precomputed cache
_rel_pos_bias_cache_7 = None


def _get_rel_pos_bias_7():
    global _rel_pos_bias_cache_7
    if _rel_pos_bias_cache_7 is None:
        N = 7
        tmp_10 = torch.arange(N, dtype=torch.int64)
        tmp_11 = tmp_10[:, None]
        tmp_12 = torch.arange(N, dtype=torch.int64)
        tmp_13 = tmp_12[None, :]
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
        tmp_19 += tmp_31
        _rel_pos_bias_cache_7 = tmp_19
    return _rel_pos_bias_cache_7


@torch.fx.wrap
def rel_pos_bias_7():
    return _get_rel_pos_bias_7()


def replacement_func():
    return rel_pos_bias_7