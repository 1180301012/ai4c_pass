import torch
import triton
import triton.language as tl


def pattern(x):
    return x.transpose(-2, -1)


def replacement_args(x):
    return (x,)


@triton.jit
def batch_transpose_kernel(
    x_ptr,
    out_ptr,
    B,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose last two dims: [B, M, N] -> [B, N, M].
    Iterate over *output* elements sequentially (coalesced writes).
    Compute the corresponding *input* offset for each (strided reads, hidden by L2 cache).
    """
    pid = tl.program_id(0)
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * N * M
    mask = out_offsets < total

    # Decompose flat output offset → (b, n, m) in [B, N, M]
    m_idx = out_offsets % M
    rem   = out_offsets // M
    n_idx = rem % N
    b_idx = rem // N

    # Compute input offset: x[b, m, n] in [B, M, N]
    in_offsets = b_idx * (M * N) + m_idx * N + n_idx

    x = tl.load(x_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, x, mask=mask)


@torch.fx.wrap
def transpose_last_two(x):
    shape = x.shape
    M = shape[-2]
    N = shape[-1]
    B = x.numel() // (M * N)

    x_contig = x.contiguous()

    out_shape = list(shape[:-2]) + [N, M]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # B*N*M = 110880 elements, BLOCK_SIZE=2048 → 55 blocks ≈ 1/SM
    BLOCK_SIZE = 2048
    total = B * N * M
    num_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    batch_transpose_kernel[(num_programs,)](
        x_contig, out,
        B, M, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return out


def replacement_func():
    return transpose_last_two