import torch


_ZERO_CACHE = {}


@torch.fx.wrap
def cached_new_zeros(example, rows, cols):
    key = (example.device, example.dtype, rows, cols)
    out = _ZERO_CACHE.get(key)
    if out is None:
        out = torch.zeros((rows, cols), device=example.device, dtype=example.dtype)
        _ZERO_CACHE[key] = out
    return out