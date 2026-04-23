import torch
import triton
import triton.language as tl


_FULL_ROUTES = {
    'src_gelu_reshape_pad',
    'aten_gelu_view_pad',
    'aten_gelu_reshape_pad',
    'erf_gelu_view_pad',
    'erf_gelu_reshape_aten_pad',
    'gelu_approx_none_reshape_pad',
}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['out_numel'],
)
@triton.jit
def _gelu_linear_kernel(
    in_ptr,
    out_ptr,
    in_numel,
    out_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = offsets < out_numel
    in_mask = offsets < in_numel
    x = tl.load(in_ptr + offsets, mask=in_mask, other=0.0).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, y, mask=out_mask)


@torch.fx.wrap
def fused_gelu_pad_dispatch(in_0, route):
    in_numel = in_0.numel()

    if route in _FULL_ROUTES:
        last_dim = 768
        out_rows = in_numel // last_dim + 1
        out_numel = out_rows * last_dim
        out = torch.empty((1, out_rows, last_dim), device=in_0.device, dtype=in_0.dtype)
        grid = lambda META: (triton.cdiv(out_numel, META['BLOCK_SIZE']),)
        _gelu_linear_kernel[grid](in_0, out, in_numel, out_numel)
        return out

    out = torch.empty_like(in_0)
    grid = lambda META: (triton.cdiv(in_numel, META['BLOCK_SIZE']),)
    _gelu_linear_kernel[grid](in_0, out, in_numel, in_numel)
    return out


def replacement_func():
    return fused_gelu_pad_dispatch