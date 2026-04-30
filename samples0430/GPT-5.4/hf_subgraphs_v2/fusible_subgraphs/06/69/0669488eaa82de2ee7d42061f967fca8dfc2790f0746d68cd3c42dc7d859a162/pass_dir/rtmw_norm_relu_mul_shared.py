import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _rtmw_relu_rmsnorm_mul_kernel(
    x_ptr,
    g_ptr,
    out_ptr,
    M,
    N,
    scale,
    eps,
    x_row_stride,
    out_row_stride,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    base = row * x_row_stride

    for start in range(0, N, BLOCK_SIZE):
        idx = start + offs
        mask = idx < N
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        x = tl.maximum(x, 0)
        xf = x.to(tl.float32)
        acc += xf * xf

    sumsq = tl.sum(acc, axis=0)
    denom = tl.maximum(tl.sqrt(sumsq) * scale, eps)
    inv = 1.0 / denom
    g = tl.load(g_ptr).to(tl.float32)
    factor = inv * g

    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, N, BLOCK_N):
        idx = start + offs_n
        mask = idx < N
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        x = tl.maximum(x, 0)
        y = x.to(tl.float32) * factor
        tl.store(out_ptr + row * out_row_stride + idx, y, mask=mask)


@torch.fx.wrap
def rtmw_relu_rmsnorm_mul(x, g, route):
    if route == 'scale_0p14433756729740643':
        scale = 0.14433756729740643
    elif route == 'scale_0p07216878364870322':
        scale = 0.07216878364870322
    else:
        raise RuntimeError(f'unknown route: {route}')

    x_contig = x.contiguous()
    orig_shape = tuple(x_contig.shape)
    if len(orig_shape) < 3:
        raise RuntimeError('expected input rank >= 3')

    M = 1
    for d in orig_shape[:-2]:
        M *= d
    N = orig_shape[-2] * orig_shape[-1]

    x2 = x_contig.view(M, N)
    out2 = torch.empty_like(x2)

    grid = lambda meta: (M,)
    _rtmw_relu_rmsnorm_mul_kernel[grid](
        x2,
        g,
        out2,
        M,
        N,
        scale,
        1.0e-5,
        x2.stride(0),
        out2.stride(0),
    )
    return out2.view(orig_shape[0], orig_shape[1], orig_shape[2] * orig_shape[3])