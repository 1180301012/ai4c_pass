import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.transpose(-2, -1)
    return (tmp_0, tmp_1)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _fused_mul_transpose_kernel(
    in0_ptr,
    in1_ptr,
    out0_ptr,
    out1_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    in0  : [BH, M, N]  -> out1 : [BH, N, M]  (transpose)
    in1  : [BH, M, N]  -> out0 : [BH, M, N]  (scalar multiply)
    BH is the product of all leading dimensions; encoded as grid dim 2.
    """
    SCALAR = 0.1767766952966369

    pid_bh = tl.program_id(2)
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m  = m_offsets < M
    mask_n  = n_offsets < N
    mask_mn = mask_m[:, None] & mask_n[None, :]   # (BLOCK_M, BLOCK_N)

    # Shared linear offset: [BH, M, N] with stride [M*N, N, 1]
    base_mn = pid_bh * M * N + m_offsets[:, None] * N + n_offsets[None, :]

    # ── Op 1: element-wise scalar multiply ─────────────────────────────────
    in1_tile = tl.load(in1_ptr + base_mn, mask=mask_mn, other=0.0)
    tl.store(out0_ptr + base_mn, in1_tile * SCALAR, mask=mask_mn)

    # ── Op 2: transpose ─────────────────────────────────────────────────────
    in0_tile = tl.load(in0_ptr + base_mn, mask=mask_mn, other=0.0)
    in0_t = tl.trans(in0_tile)                     # (BLOCK_N, BLOCK_M)

    mask_nm = mask_n[:, None] & mask_m[None, :]   # (BLOCK_N, BLOCK_M)
    base_nm = pid_bh * N * M + n_offsets[:, None] * M + m_offsets[None, :]
    tl.store(out1_ptr + base_nm, in0_t, mask=mask_nm)


@torch.fx.wrap
def fused_mul_transpose(in_0, in_1):
    # in_0: tensor to be transposed   -> out1
    # in_1: tensor to be scaled       -> out0
    B  = in_0.shape[0]
    H  = in_0.shape[1]
    M  = in_0.shape[2]
    N  = in_0.shape[3]
    BH = B * H

    out0 = torch.empty_like(in_1)                                    # scaled output
    out1 = torch.empty(B, H, N, M, dtype=in_0.dtype, device=in_0.device)  # transposed output

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        BH,
    )
    _fused_mul_transpose_kernel[grid](in_0, in_1, out0, out1, M, N, BLOCK_M, BLOCK_N)
    return out0, out1


def replacement_func():
    return fused_mul_transpose