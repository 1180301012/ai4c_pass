import math
import torch

# Precompute the N=45 relative position bias at module load time using pure Python.
_N = 45
_LOG16 = math.log(16.0)
_DATA_45 = []
for _i in range(_N):
    _row = []
    for _j in range(_N):
        _d = _i - _j
        _sign = 16 if _d < 0 else 0
        _abs_d = abs(_d)
        if _abs_d < 8:
            _bucket = _abs_d
        else:
            _bucket = min(8 + int(8.0 * math.log(_abs_d / 8.0) / _LOG16), 15)
        _row.append(_sign + _bucket)
    _DATA_45.append(_row)


@torch.fx.wrap
def rel_pos_bias_45():
    return torch.as_tensor(_DATA_45, dtype=torch.int64)


def pattern():
    tmp_10 = torch.arange(45, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(45, dtype=torch.int64)
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


def replacement_func():
    return rel_pos_bias_45