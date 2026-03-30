import torch
import triton
import triton.language as tl


@triton.jit
def _coat_slice_transpose_kernel(
    in2_ptr, out_ptr,
    Heads, N, D, N_out,
    TILE_N: tl.constexpr,
    TILE_D: tl.constexpr,
):
    """
    Fused slice[:, :, 1:, :] + transpose(-1, -2) for [1, Heads, N, D].
    Output: [1, Heads, D, N_out] contiguous, N_out = N - 1.

    2D tiling for coalesced reads (D fast) and coalesced writes (N_out fast
    after transpose). Grid: (Heads, cdiv(N_out, TILE_N), cdiv(D, TILE_D)).
    """
    pid_head = tl.program_id(0)
    pid_n    = tl.program_id(1)
    pid_d    = tl.program_id(2)

    n_start = pid_n * TILE_N
    d_start = pid_d * TILE_D

    n_offs = n_start + tl.arange(0, TILE_N)   # [TILE_N]
    d_offs = d_start + tl.arange(0, TILE_D)   # [TILE_D]

    n_mask = n_offs < N_out
    d_mask = d_offs < D

    # Coalesced READ: d varies fast in memory => shape [TILE_N, TILE_D]
    in_ptrs = (in2_ptr
               + pid_head * (N * D)
               + (n_offs[:, None] + 1) * D
               + d_offs[None, :])
    tile = tl.load(in_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    # tile: [TILE_N, TILE_D]

    # Coalesced WRITE: n varies fast in memory => shape [TILE_D, TILE_N]
    out_ptrs = (out_ptr
                + pid_head * (D * N_out)
                + d_offs[:, None] * N_out
                + n_offs[None, :])
    tl.store(out_ptrs, tl.trans(tile), mask=d_mask[:, None] & n_mask[None, :])


def pattern(in_2):
    tmp_2 = in_2[:, :, 1:, :]
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3


def replacement_args(in_2):
    return (in_2,)


@torch.fx.wrap
def _coat_fused_slice_transpose(in_2):
    _B, Heads, N, D = in_2.shape
    N_out = N - 1
    out = torch.empty(1, Heads, D, N_out, dtype=in_2.dtype, device=in_2.device)

    TILE_N = 16
    TILE_D = 16
    grid = (
        Heads,
        triton.cdiv(N_out, TILE_N),
        triton.cdiv(D, TILE_D),
    )
    _coat_slice_transpose_kernel[grid](
        in_2, out, Heads, N, D, N_out,
        TILE_N=TILE_N, TILE_D=TILE_D,
        num_warps=2,
    )

    return out


def replacement_func():
    return _coat_fused_slice_transpose