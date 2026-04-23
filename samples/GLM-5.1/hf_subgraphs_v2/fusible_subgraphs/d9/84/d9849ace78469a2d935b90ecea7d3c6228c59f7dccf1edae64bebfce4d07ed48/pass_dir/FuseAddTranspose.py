import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    in_1 += in_0
    tmp_2 = in_1.transpose(1, 2)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_transpose_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, C, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # pid corresponds to a tile in the output [B, N, C] layout
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    # Output layout is [B, N, C] (transposed from [B, C, N])
    # We iterate over B dimension inside the kernel
    for b in range(B):
        # in_1 is [B, C, N]: in_1[b, m, n] = in_1_ptr[b * C * N + m * N + n]
        # in_0 is [C, 1]: in_0[m, 0] = in_0_ptr[m * 1 + 0] = in_0_ptr[m]
        # out is [B, N, C]: out[b, n, m] = out_ptr[b * N * C + n * C + m]

        in_1_ptrs = in_1_ptr + b * C * N + m_offsets[:, None] * N + n_offsets[None, :]
        mask_1 = (m_offsets[:, None] < C) & (n_offsets[None, :] < N)

        in_0_ptrs = in_0_ptr + m_offsets[:, None]
        mask_0 = m_offsets[:, None] < C

        in_1_vals = tl.load(in_1_ptrs, mask=mask_1, other=0.0)
        in_0_vals = tl.load(in_0_ptrs, mask=mask_0, other=0.0)

        result = in_1_vals + in_0_vals

        out_ptrs = out_ptr + b * N * C + n_offsets[None, :] * C + m_offsets[:, None]
        mask_out = (m_offsets[:, None] < C) & (n_offsets[None, :] < N)
        tl.store(out_ptrs, result, mask=mask_out)


@torch.fx.wrap
def fused_add_transpose(in_0, in_1):
    # in_0: [C, 1] bias, in_1: [B, C, N]
    B = in_1.shape[0]
    C = in_1.shape[1]
    N = in_1.shape[2]

    # Output: [B, N, C]
    out = torch.empty((B, N, C), dtype=in_1.dtype, device=in_1.device)

    BLOCK_M = 32
    BLOCK_N = 32

    grid_m = (C + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    fused_add_transpose_kernel[(grid_m, grid_n)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B, C=C, N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out


def replacement_func():
    return fused_add_transpose