import torch
import triton
import triton.language as tl
import numpy as np
import operator

# Precompute the deterministic index table for N=14 at module load time (CPU, using numpy)
def _compute_index_data_14():
    N = 14
    r = np.arange(N)
    c = np.arange(N)
    R, C = np.meshgrid(r, c, indexing='ij')
    flat_r = R.ravel()  # [N*N]
    flat_c = C.ravel()  # [N*N]
    M = N * N
    dr = (flat_r[:, None] - flat_r[None, :] + (N - 1)).astype(np.int64)  # [M, M]
    dc = (flat_c[:, None] - flat_c[None, :] + (N - 1)).astype(np.int64)  # [M, M]
    idx = dr * (2 * N - 1) + dc  # [M, M]
    result = np.zeros((M + 1, M + 1), dtype=np.int64)
    result[1:, 1:] = idx
    result[0, :] = (2 * N - 1) ** 2        # 729
    result[:, 0] = (2 * N - 1) ** 2 + 1    # 730
    result[0, 0] = (2 * N - 1) ** 2 + 2    # 731
    return result.ravel()

_INDEX_DATA_14 = _compute_index_data_14()
_INDEX_GPU_CACHE_14 = {}


@triton.jit
def _triton_cat_2d_kernel_14(
    a_ptr, b_ptr, out_ptr,
    N_a, N_b, K,
    BLOCK: tl.constexpr,
):
    """Concatenate two 2D tensors along dim=0: out = [a; b]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    total = (N_a + N_b) * K
    mask = offsets < total

    row = offsets // K
    col = offsets % K
    is_a = row < N_a

    a_row = tl.where(is_a, row, 0)
    b_row = tl.where(is_a, 0, row - N_a)

    a_val = tl.load(a_ptr + a_row * K + col, mask=(mask & is_a), other=0.0)
    b_val = tl.load(b_ptr + b_row * K + col, mask=(mask & ~is_a), other=0.0)

    out_val = tl.where(is_a, a_val, b_val)
    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def _beit_triton_cat_n14(in_0, in_1):
    N_a = in_1.shape[0]
    N_b = in_0.shape[0]
    K = in_1.shape[1]
    out_cat = torch.empty((N_a + N_b, K), dtype=in_1.dtype, device=in_1.device)
    BLOCK = 512
    num_progs = (out_cat.numel() + BLOCK - 1) // BLOCK
    _triton_cat_2d_kernel_14[(num_progs,)](
        in_1, in_0, out_cat,
        N_a, N_b, K,
        BLOCK=BLOCK,
    )
    return out_cat


# ── Pattern & plumbing ──────────────────────────────────────────────────────

def pattern(in_0, in_1):
    return torch.cat([in_1, in_0])


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _beit_triton_cat_n14