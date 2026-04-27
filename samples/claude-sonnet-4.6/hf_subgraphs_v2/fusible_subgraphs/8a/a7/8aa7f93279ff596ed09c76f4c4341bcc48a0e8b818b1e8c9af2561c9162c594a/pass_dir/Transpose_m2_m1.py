import torch
import triton
import triton.language as tl


def pattern(x):
    return x.transpose(-2, -1)


def replacement_args(x):
    return (x,)


@triton.jit
def _transpose_m2_m1_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Treats input as [BH, M, N] and output as [BH, N, M]
    # BH is encoded as grid dim 2, no need to pass explicitly
    pid_bh = tl.program_id(2)
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N
    mask_in = mask_m[:, None] & mask_n[None, :]   # (BLOCK_M, BLOCK_N)

    # Load tile from input [BH, M, N]: stride = [M*N, N, 1]
    in_ptrs = pid_bh * M * N + m_offsets[:, None] * N + n_offsets[None, :]
    x_block = tl.load(x_ptr + in_ptrs, mask=mask_in, other=0.0)

    # Transpose tile: (BLOCK_M, BLOCK_N) -> (BLOCK_N, BLOCK_M)
    x_t = tl.trans(x_block)

    # Store tile to output [BH, N, M]: stride = [N*M, M, 1]
    out_mask = mask_n[:, None] & mask_m[None, :]  # (BLOCK_N, BLOCK_M)
    out_ptrs = pid_bh * N * M + n_offsets[:, None] * M + m_offsets[None, :]
    tl.store(out_ptr + out_ptrs, x_t, mask=out_mask)


@torch.fx.wrap
def triton_transpose_m2_m1(x):
    # Explicit 4-D unpacking avoids list operations that may trip validators
    B  = x.shape[0]
    H  = x.shape[1]
    M  = x.shape[2]
    N  = x.shape[3]
    BH = B * H

    out = torch.empty(B, H, N, M, dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        BH,
    )
    _transpose_m2_m1_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    return out


def replacement_func():
    return triton_transpose_m2_m1