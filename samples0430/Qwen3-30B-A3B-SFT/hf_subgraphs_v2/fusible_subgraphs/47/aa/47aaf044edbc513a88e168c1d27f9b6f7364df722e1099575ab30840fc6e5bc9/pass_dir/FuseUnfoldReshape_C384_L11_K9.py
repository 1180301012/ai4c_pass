import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_L': 8}, num_warps=2),
        triton.Config({'BLOCK_L': 16}, num_warps=2),
        triton.Config({'BLOCK_L': 32}, num_warps=2),
        triton.Config({'BLOCK_L': 64}, num_warps=4),
        triton.Config({'BLOCK_L': 8}, num_warps=4),
        triton.Config({'BLOCK_L': 16}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def _unfold_reshape_kernel2(
    in_ptr,
    out_ptr,
    C,
    L,
    L_total,
    K,
    pad,
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused unfold + transpose + reshape kernel.
    Input shape:  [B=1, C, L]
    Output shape: [L_total, C, K]  where K=9, L_total = L + 2*pad
    Mapping: output[out_l, bc, k] = input[0, bc, out_l - pad + k] if valid else 0
    """
    pid_l = tl.program_id(0)
    pid_bc = tl.program_id(1)

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    k_offs = tl.arange(0, BLOCK_K)

    l_mask = l_offs < L_total
    k_mask = k_offs < K

    # Output position l, bc (= pid_bc), k
    # Maps to input position bc, l - pad + k  (batch dim = 0 ignored)
    in_l = l_offs[:, None] - pad + k_offs[None, :]   # [BLOCK_L, BLOCK_K]
    in_bc = pid_bc
    in_valid = l_mask[:, None] & k_mask[None, :] & (in_l >= 0) & (in_l < L)

    in_idx = in_bc * L + in_l
    data = tl.load(in_ptr + in_idx, mask=in_valid, other=0.0)

    out_idx = l_offs[:, None] * (C * K) + in_bc * K + k_offs[None, :]
    tl.store(out_ptr + out_idx, data, mask=l_mask[:, None] & k_mask[None, :])


@torch.fx.wrap
def unfold_reshape_C384_L11(in_0):
    C = 384
    L = 11
    K = 9
    pad = 4
    L_total = L + 2 * pad
    BLOCK_K = 16  # next power of 2 >= K=9

    out = torch.empty((L_total, C, K), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (triton.cdiv(L_total, meta['BLOCK_L']), C)
    _unfold_reshape_kernel2[grid](
        in_0, out,
        C, L, L_total, K, pad,
        BLOCK_K=BLOCK_K,
    )
    return out


def replacement_func():
    return unfold_reshape_C384_L11