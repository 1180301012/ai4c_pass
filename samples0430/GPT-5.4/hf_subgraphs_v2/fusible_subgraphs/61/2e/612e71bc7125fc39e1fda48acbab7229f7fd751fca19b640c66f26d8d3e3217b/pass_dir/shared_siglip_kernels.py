import torch
import triton
import triton.language as tl
from torch import device


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=1),
        triton.Config({'BLOCK': 256}, num_warps=1),
        triton.Config({'BLOCK': 256}, num_warps=2),
        triton.Config({'BLOCK': 512}, num_warps=2),
    ],
    key=['D'],
)
@triton.jit
def l2norm2rows_kernel(
    x_ptr,
    out_ptr,
    D,
    x_stride_0,
    x_stride_1,
    out_stride_0,
    out_stride_1,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_x_ptr = x_ptr + row * x_stride_0
    row_out_ptr = out_ptr + row * out_stride_0

    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, 2048, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(row_x_ptr + offs * x_stride_1, mask=mask, other=0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)

    inv_norm = 1.0 / tl.sqrt(acc)

    for start in range(0, 2048, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(row_x_ptr + offs * x_stride_1, mask=mask, other=0).to(tl.float32)
        y = x * inv_norm
        tl.store(row_out_ptr + offs * out_stride_1, y.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=1),
        triton.Config({'BLOCK': 256}, num_warps=1),
        triton.Config({'BLOCK': 512}, num_warps=2),
        triton.Config({'BLOCK': 1024}, num_warps=2),
    ],
    key=['D'],
)
@triton.jit
def transpose_1xd_to_dx1_kernel(
    x_ptr,
    out_ptr,
    D,
    x_stride_0,
    x_stride_1,
    out_stride_0,
    out_stride_1,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + offs * x_stride_1, mask=mask, other=0)
    tl.store(out_ptr + offs * out_stride_0, x, mask=mask)


@torch.fx.wrap
def shared_siglip_dispatch(*args):
    route = args[-1]

    if route == 'norm':
        in_1 = args[0]
        D = in_1.shape[-1]
        out = torch.empty_like(in_1)
        l2norm2rows_kernel[(2,)](
            in_1,
            out,
            D,
            in_1.stride(0),
            in_1.stride(1),
            out.stride(0),
            out.stride(1),
        )
        return out

    if route == 'transpose':
        in_0 = args[0]
        D = in_0.shape[1]
        out = torch.empty((D, 1), device=in_0.device, dtype=in_0.dtype)
        grid = lambda META: (triton.cdiv(D, META['BLOCK']),)
        transpose_1xd_to_dx1_kernel[grid](
            in_0,
            out,
            D,
            in_0.stride(0),
            in_0.stride(1),
            out.stride(0),
            out.stride(1),
        )
        return out

    raise ValueError('Unknown route')


def shared_replacement_func():
    return shared_siglip_dispatch