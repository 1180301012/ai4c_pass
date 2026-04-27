import torch
import triton
import triton.language as tl

_REGISTRY = {}


# ─── Triton kernel (autotune picks the best block size during warmup) ───────────
# out[l*C9 + c_k]  =  in_0[0, c_k, l]   (in_0 strides: [C9*L, L, 1])
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
    ],
    key=['C9', 'L'],
)
@triton.jit
def _post_unfold_kernel(
    in0_ptr,
    out_ptr,
    C9,
    L,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    c_k = offs % C9
    l   = offs // C9
    x = tl.load(in0_ptr + c_k * L + l, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def _run_route_post_16_8(in_0):
    C9 = in_0.shape[1]
    L  = in_0.shape[2]
    group_size = 8
    total_rows = L * (C9 // (group_size * 9))
    total_elements = total_rows * group_size * 9
    out = torch.empty((total_rows, group_size, 9), dtype=in_0.dtype, device=in_0.device)
    _post_unfold_kernel[
        lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    ](in_0, out, C9, L, total_elements)
    return out


def _run_route_post_384_64(in_0):
    C9 = in_0.shape[1]
    L  = in_0.shape[2]
    group_size = 64
    total_rows = L * (C9 // (group_size * 9))
    total_elements = total_rows * group_size * 9
    out = torch.empty((total_rows, group_size, 9), dtype=in_0.dtype, device=in_0.device)
    _post_unfold_kernel[
        lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    ](in_0, out, C9, L, total_elements)
    return out


_REGISTRY["route_post_16_8"]  = _run_route_post_16_8
_REGISTRY["route_post_384_64"] = _run_route_post_384_64


# ─── Shared @torch.fx.wrap – SAME OBJECT imported by both pass files ────────────
@torch.fx.wrap
def convbert_dispatch(in_0, route):
    fn = _REGISTRY.get(route)
    if fn is not None:
        return fn(in_0)
    return in_0