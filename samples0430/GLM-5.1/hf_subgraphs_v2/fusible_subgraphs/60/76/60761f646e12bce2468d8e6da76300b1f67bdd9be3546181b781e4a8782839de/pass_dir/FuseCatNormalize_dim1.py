import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return (tmp_1,)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def l2_normalize_kernel(
    x_ptr, out_ptr,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    row_start = row_id * N

    # Compute sum of squares for this row using float32 accumulation
    sum_sq = 0.0
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        sum_sq = sum_sq + tl.sum(x.to(tl.float32) * x.to(tl.float32))

    # Compute norm = max(sqrt(sum_sq), eps)
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, eps)
    inv_norm = 1.0 / norm

    # Normalize: multiply each element by inv_norm
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        out = x.to(tl.float32) * inv_norm
        tl.store(out_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def l2_normalize(x):
    M, N = x.shape
    eps = 1e-12

    out = torch.empty_like(x)

    grid = (M,)
    l2_normalize_kernel[grid](
        x_ptr=x, out_ptr=out,
        M=M, N=N,
        eps=eps,
    )

    return out


def replacement_func():
    return l2_normalize