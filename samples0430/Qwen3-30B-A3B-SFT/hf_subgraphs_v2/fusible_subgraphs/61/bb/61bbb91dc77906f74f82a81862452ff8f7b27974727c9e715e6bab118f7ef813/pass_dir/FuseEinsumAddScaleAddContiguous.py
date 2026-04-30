import torch
import triton
import triton.language as tl


def pattern(in_0, in_2, einsum_result):
    # einsum_result = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    tmp_3 = einsum_result * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


def replacement_args(in_0, in_2, einsum_result):
    return (in_0, in_2, einsum_result)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def _fused_einsum_add_scale_add_kernel(
    einsum_ptr, in2_ptr, out_ptr, in0_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    e     = tl.load(einsum_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    i     = tl.load(in2_ptr    + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(in0_ptr)

    result = e * scale + i

    tl.store(out_ptr + offs, result, mask=mask)


@torch.fx.wrap
def fused_einsum_add_scale_add(in_0, in_2, einsum_result):
    N   = einsum_result.numel()
    out = torch.empty_like(einsum_result)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)

    _fused_einsum_add_scale_add_kernel[grid](
        einsum_result, in_2, out, in_0,
        N,
    )

    return out


def replacement_func():
    return fused_einsum_add_scale_add