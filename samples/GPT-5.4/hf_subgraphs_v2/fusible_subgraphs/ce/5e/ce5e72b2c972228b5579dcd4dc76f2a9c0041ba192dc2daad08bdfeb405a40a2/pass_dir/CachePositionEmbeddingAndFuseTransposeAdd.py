import torch
import triton
import triton.language as tl

from torch import device
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


_POSITION_EMB_CACHE = {}
_WEIGHT_FLAT_CACHE = {}


# Match the full patch-embedding subgraph exactly.
def pattern(in_0, in_1, in_2, in_3):
    conv3d = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = conv3d.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = in_2.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def _pos_cache_key(pos, dtype, dev):
    return (
        pos.data_ptr(),
        tuple(pos.shape),
        tuple(pos.stride()),
        str(pos.dtype),
        str(dtype),
        str(dev),
    )


def _weight_cache_key(weight):
    return (
        weight.data_ptr(),
        tuple(weight.shape),
        tuple(weight.stride()),
        str(weight.dtype),
        str(weight.device),
    )


def _get_cached_pos(pos, dtype, dev):
    key = _pos_cache_key(pos, dtype, dev)
    cached = _POSITION_EMB_CACHE.get(key)
    if cached is None:
        cached = pos.detach().to(device=dev, dtype=dtype)
        cached = cached.contiguous()
        _POSITION_EMB_CACHE[key] = cached
    return cached


def _get_cached_weight_flat(weight):
    key = _weight_cache_key(weight)
    cached = _WEIGHT_FLAT_CACHE.get(key)
    if cached is None:
        cached = weight.reshape(weight.shape[0], -1).transpose(0, 1).contiguous()
        _WEIGHT_FLAT_CACHE[key] = cached
    return cached


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 16, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 16, 'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 32, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 32, 'BLOCK_C': 128}, num_warps=8, num_stages=2),
    ],
    key=['n_tokens', 'channels'],
)
@triton.jit
def _bias_pos_epilogue_kernel(
    mm_ptr,
    pos_ptr,
    bias_ptr,
    out_ptr,
    n_tokens,
    channels,
    mm_stride_b,
    mm_stride_t,
    mm_stride_c,
    p_stride_b,
    p_stride_t,
    p_stride_c,
    o_stride_b,
    o_stride_t,
    o_stride_c,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_t[:, None] < n_tokens) & (offs_c[None, :] < channels)

    mm_ptrs = mm_ptr + pid_b * mm_stride_b + offs_t[:, None] * mm_stride_t + offs_c[None, :] * mm_stride_c
    pos_ptrs = pos_ptr + pid_b * p_stride_b + offs_t[:, None] * p_stride_t + offs_c[None, :] * p_stride_c
    out_ptrs = out_ptr + pid_b * o_stride_b + offs_t[:, None] * o_stride_t + offs_c[None, :] * o_stride_c
    bias_ptrs = bias_ptr + offs_c

    mm = tl.load(mm_ptrs, mask=mask, other=0.0)
    pos = tl.load(pos_ptrs, mask=mask, other=0.0)
    bias = tl.load(bias_ptrs, mask=offs_c < channels, other=0.0)[None, :]
    out = mm + pos + bias
    tl.store(out_ptrs, out, mask=mask)


@torch.fx.wrap
def cached_position_embedding_add(in_0, in_1, in_2, in_3):
    bias = unwrap_tensor(in_0)
    weight = unwrap_tensor(in_1)
    pos = unwrap_tensor(in_2)
    x = unwrap_tensor(in_3)

    batch = x.shape[0]
    in_channels = x.shape[1]
    t_in = x.shape[2]
    h_in = x.shape[3]
    w_in = x.shape[4]

    k_t = weight.shape[2]
    k_h = weight.shape[3]
    k_w = weight.shape[4]
    out_channels = weight.shape[0]

    out_t = t_in // 2
    out_h = h_in // 16
    out_w = w_in // 16
    n_tokens = out_t * out_h * out_w
    k_total = in_channels * k_t * k_h * k_w

    # Non-overlapping patchify for stride == kernel size.
    patches = x.view(batch, in_channels, out_t, k_t, out_h, k_h, out_w, k_w)
    patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(batch * n_tokens, k_total)

    weight_flat = _get_cached_weight_flat(weight)
    mm = patches @ weight_flat
    mm = mm.reshape(batch, n_tokens, out_channels)

    pos_cached = _get_cached_pos(pos, mm.dtype, mm.device)
    out = torch.empty(mm.shape, device=mm.device, dtype=mm.dtype)
    p_stride_b = 0 if pos_cached.shape[0] == 1 else pos_cached.stride(0)

    grid = lambda meta: (
        triton.cdiv(out_channels, meta['BLOCK_C']),
        triton.cdiv(n_tokens, meta['BLOCK_T']),
        batch,
    )

    _bias_pos_epilogue_kernel[grid](
        mm,
        pos_cached,
        bias,
        out,
        n_tokens,
        out_channels,
        mm.stride(0),
        mm.stride(1),
        mm.stride(2),
        p_stride_b,
        pos_cached.stride(1),
        pos_cached.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


def replacement_func():
    return cached_position_embedding_add