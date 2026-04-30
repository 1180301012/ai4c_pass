import torch


_MUL_CACHE = {}
_ZERO_CACHE = {}


@torch.fx.wrap
def cached_full_graph_leaf(in_0, in_1, in_2, zero_rows, zero_cols):
    mul_key = (tuple(in_2.shape), in_2.dtype, in_2.device)
    out_mul = _MUL_CACHE.get(mul_key)
    if out_mul is None:
        out_mul = torch.empty_like(in_2)
        _MUL_CACHE[mul_key] = out_mul

    zero_key = (in_2.device, in_2.dtype, zero_rows, zero_cols)
    out_zero = _ZERO_CACHE.get(zero_key)
    if out_zero is None:
        out_zero = torch.zeros((zero_rows, zero_cols), device=in_2.device, dtype=in_2.dtype)
        _ZERO_CACHE[zero_key] = out_zero

    out_mul.copy_(in_2)
    out_mul.mul_(in_1.view(-1, 1))
    out_expand = in_0.view((-1, 1)).expand_as(out_mul)
    return out_expand, out_zero, out_mul


def cached_full_graph_wrapper(in_0, in_1, in_2, zero_rows, zero_cols):
    out_expand, out_zero, out_mul = cached_full_graph_leaf(in_0, in_1, in_2, zero_rows, zero_cols)
    return out_expand, out_zero, out_mul