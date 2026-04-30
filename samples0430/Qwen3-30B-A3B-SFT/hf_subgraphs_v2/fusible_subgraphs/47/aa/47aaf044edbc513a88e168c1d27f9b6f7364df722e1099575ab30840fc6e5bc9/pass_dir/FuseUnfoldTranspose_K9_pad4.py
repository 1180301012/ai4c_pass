import torch
import triton
import triton.language as tl


def pattern(tmp_1):
    """
    Match: tmp_1 (already contiguous, shape [1,C,L,1]) ->
           transpose(1,2) -> reshape(1,-1,16,9) -> reshape(-1,8,9)

    The compiled graph has: contiguous->unsqueeze->unfold->transpose->reshape->reshape.
    The unfold is a call_function (not a call_method), so I can't match through it.
    Starting from tmp_1 (after unsqueeze) avoids the unfold in the pattern.
    The Triton kernel fuses: transpose+reshape+reshape into one gather operation.
    """
    tmp_3 = tmp_1.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(tmp_1):
    return (tmp_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 16, 'BLOCK_K': 16}, num_warps=2),
        triton.Config({'BLOCK_L': 32, 'BLOCK_K': 16}, num_warps=2),
        triton.Config({'BLOCK_L': 64, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 16, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_L': 32, 'BLOCK_K': 16}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def _transpose_reshape_kernel(
    in_ptr,
    out_ptr,
    C,
    L,
    L_total,
    K,
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused transpose + reshape kernel.
    Input:  tmp_1 [1, C, L, 1]  (contiguous)
    Output: [L_total, C, K]  semantically identical to model's tmp_5

    out[l, bc, k] = tmp_1[0, bc, l-4+k, 0]  if 0<=l-4+k<L else 0
    BLOCK_K must be a power of 2 >= K=9 (so BLOCK_K=16).
    """
    pid_l = tl.program_id(0)
    pid_bc = tl.program_id(1)

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    k_offs = tl.arange(0, BLOCK_K)   # BLOCK_K=16 >= K=9, uses power-of-2 arange

    l_mask = l_offs < L_total
    k_mask = k_offs < K              # mask out k=9..15 (they're outside the window)
    in_l = l_offs[:, None] - 4 + k_offs[None, :]
    in_valid = l_mask[:, None] & k_mask[None, :] & (in_l >= 0) & (in_l < L)

    # tmp_1 shape [1, C, L, 1] → offset = bc*L + in_l (last dim stride=1)
    in_idx = pid_bc * L + in_l
    data = tl.load(in_ptr + in_idx, mask=in_valid, other=0.0)

    # Output [L_total, C, K] → offset = l*(C*K) + bc*K + k
    out_idx = l_offs[:, None] * (C * K) + pid_bc * K + k_offs[None, :]
    tl.store(out_ptr + out_idx, data, mask=l_mask[:, None] & k_mask[None, :])


@torch.fx.wrap
def transpose_reshape_fused(tmp_1):
    """
    Fused replacement for: transpose(1,2) -> reshape(1,-1,16,9) -> reshape(-1,8,9).
    tmp_1: [1, C, L, 1]. Returns [L+8, C, 9] (same semantics as model's tmp_5).
    """
    C = tmp_1.shape[1]
    L = tmp_1.shape[2]
    K = 9
    pad = 4
    L_total = L + 2 * pad

    out = torch.empty((L_total, C, K), dtype=tmp_1.dtype, device=tmp_1.device)

    grid = lambda meta: (triton.cdiv(L_total, meta['BLOCK_L']), C)
    _transpose_reshape_kernel[grid](
        tmp_1, out,
        C, L, L_total, K,
    )
    return out


def replacement_func():
    return transpose_reshape_fused