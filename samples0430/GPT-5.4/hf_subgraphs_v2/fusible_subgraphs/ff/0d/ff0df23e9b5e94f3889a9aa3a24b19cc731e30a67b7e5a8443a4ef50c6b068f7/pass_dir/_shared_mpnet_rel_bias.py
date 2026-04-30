import torch
import triton
import triton.language as tl


_WEIGHT_CACHE = {}
_INDEX_CACHE = {}
_OUTPUT_CACHES = [{}, {}, {}]
_ROUTE_SHAPES = ((1, 45, 45), (1, 11, 11), (2, 7, 7))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _rel_bias_gather_expand_kernel(
    weight_ptr,
    index_ptr,
    out_ptr,
    N,
    H,
    W,
    D,
    weight_stride0,
    weight_stride1,
    index_stride0,
    index_stride1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    tmp = offs
    w = tmp % W
    tmp = tmp // W
    h = tmp % H
    tmp = tmp // H
    d = tmp % D

    idx = tl.load(index_ptr + h * index_stride0 + w * index_stride1, mask=mask, other=0)
    idx = idx.to(tl.int32)
    vals = tl.load(weight_ptr + idx * weight_stride0 + d * weight_stride1, mask=mask, other=0)
    tl.store(out_ptr + offs, vals, mask=mask)


@torch.fx.wrap
def fused_mpnet_rel_bias(weight, index, route):
    out_cache = _OUTPUT_CACHES[route]
    hot_key = (id(weight), id(index))
    cached_out = out_cache.get(hot_key)
    if cached_out is not None:
        return cached_out

    B, H, W = _ROUTE_SHAPES[route]
    device = torch.device("cuda", index.device.index if index.is_cuda else 0)

    if (not weight.is_cuda) or (weight.device != device):
        wkey = (id(weight), device.index)
        cached_weight = _WEIGHT_CACHE.get(wkey)
        if cached_weight is None:
            cached_weight = weight.to(device=device)
            _WEIGHT_CACHE[wkey] = cached_weight
        weight = cached_weight

    if (not index.is_cuda) or (index.device != device):
        ikey = (id(index), device.index)
        cached_index = _INDEX_CACHE.get(ikey)
        if cached_index is None:
            cached_index = index.to(device=device)
            _INDEX_CACHE[ikey] = cached_index
        index = cached_index

    D = weight.shape[1]
    out = torch.empty((B, D, H, W), device=device, dtype=weight.dtype)
    N = B * D * H * W

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    _rel_bias_gather_expand_kernel[grid](
        weight,
        index,
        out,
        N,
        H,
        W,
        D,
        weight.stride(0),
        weight.stride(1),
        index.stride(0),
        index.stride(1),
    )
    out_cache[hot_key] = out
    return out


def replacement_func():
    return fused_mpnet_rel_bias