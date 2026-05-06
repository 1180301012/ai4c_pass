import torch
import triton
import triton.language as tl


def pattern(in_0):
    t = in_0.unsqueeze(1)
    out = t.transpose(2, 3)
    return out


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}),
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['N', 'in_n', 'in_d'],
)
@triton.jit
def _unsqueeze_transpose_kernel(
    in_ptr,
    out_ptr,
    N,
    in_n,
    in_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Output layout: [1, 1, in_d, in_n] (contiguous)
    # Each output element maps to in[b, n, d] = in[0, n, d] for B=1
    # Input[n * in_d + d] goes to out[d * in_n + n]
    n_idx = offsets % in_n   # innermost dimension index
    d_idx = (offsets // in_n) % in_d  # middle dimension index (Squeeze dim=1, B=1)

    in_offsets = n_idx * in_d  + d_idx
    x = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def _unsqueeze_transpose(in_0):
    B = in_0.shape[0]
    in_n = in_0.shape[1]
    in_d = in_0.shape[2]

    # Output shape: [B, 1, in_d, in_n]
    out = torch.empty((B, 1, in_d, in_n), dtype=in_0.dtype, device=in_0.device)

    N = B * in_d * in_n

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    _unsqueeze_transpose_kernel[grid](
        in_0,
        out,
        N,
        in_n,
        in_d,
    )

    return out


def replacement_func():
    return _unsqueeze_transpose