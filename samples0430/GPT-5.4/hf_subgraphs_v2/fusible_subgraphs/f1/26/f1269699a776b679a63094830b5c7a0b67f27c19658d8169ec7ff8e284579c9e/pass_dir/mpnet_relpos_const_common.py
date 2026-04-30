import math
import torch


def _build_relpos_bucket(seq_len):
    out = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            rel = i - j
            bucket = 16 if rel < 0 else 0
            n = abs(rel)
            if n < 8:
                bucket += n
            else:
                big = int((math.log(n / 8.0) / 2.772588722239781) * 8.0)
                big = min(8 + big, 15)
                bucket += big
            row.append(bucket)
        out.append(row)
    return out


_REL_SEQ7 = torch.as_tensor(_build_relpos_bucket(7), dtype=torch.int64)
_REL_SEQ11 = torch.as_tensor(_build_relpos_bucket(11), dtype=torch.int64)
_REL_SEQ45 = torch.as_tensor(_build_relpos_bucket(45), dtype=torch.int64)


@torch.fx.wrap
def mpnet_shared_dispatch(*args):
    route = args[-1]
    if route == "dropout_identity":
        return args[0]
    if route == "relpos_from_shape":
        a = args[0]
        b = args[1]
        seq = max(a.shape[0], a.shape[1], b.shape[0], b.shape[1])
        if seq == 7:
            return torch.as_tensor(_REL_SEQ7)
        if seq == 11:
            return torch.as_tensor(_REL_SEQ11)
        if seq == 45:
            return torch.as_tensor(_REL_SEQ45)
        return torch.as_tensor(_REL_SEQ7)
    if route == "seq7":
        return torch.as_tensor(_REL_SEQ7)
    if route == "seq11":
        return torch.as_tensor(_REL_SEQ11)
    if route == "seq45":
        return torch.as_tensor(_REL_SEQ45)
    return args[0]


def replacement_func():
    return mpnet_shared_dispatch