import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['B', 'H', 'N', 'D'],
)
@triton.jit
def _matmul_permute_kernel(
    attn_ptr,   # [B, H, N, N] contiguous - attention weights
    value_ptr,  # [B, H, N, D] contiguous - value
    out_ptr,    # [B, N, H, D] contiguous - output in permuted layout
    B, H, N, D,
    N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # PID decomposition: i varies fastest → all N rows of same (b,h) are consecutive
    # → value matrix [N,D] stays in L2 cache across the N programs of same (b,h)
    pid = tl.program_id(0)
    i = pid % N
    h = (pid // N) % H
    b = pid // (N * H)

    k_range = tl.arange(0, N_BLOCK)
    k_mask = k_range < N

    # Load attention weights row: attn[b, h, i, :]
    # Stride: [H*N*N, N*N, N, 1]
    attn_base = b * H * N * N + h * N * N + i * N
    attn_w = tl.load(attn_ptr + attn_base + k_range, mask=k_mask, other=0.0).to(tl.float32)

    # Load value matrix: value[b, h, :, :]
    # Stride: [H*N*D, N*D, D, 1]
    d_range = tl.arange(0, D_BLOCK)
    d_mask = d_range < D

    v_base = b * H * N * D + h * N * D
    v_2d = tl.load(
        value_ptr + v_base + k_range[:, None] * D + d_range[None, :],
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float32)

    # Weighted sum: result[d] = sum_k attn_w[k] * value[b,h,k,d]
    acc = tl.sum(attn_w[:, None] * v_2d, axis=0)  # [D_BLOCK]

    # Write directly in permuted [B, N, H, D] layout
    # Consecutive h values → consecutive D-blocks in memory (coalesced)
    out_base = b * N * H * D + i * H * D + h * D
    if IS_BF16:
        tl.store(out_ptr + out_base + d_range, acc.to(tl.bfloat16), mask=d_mask)
    else:
        tl.store(out_ptr + out_base + d_range, acc.to(tl.float16), mask=d_mask)


@torch.fx.wrap
def _matmul_permute_wrapper(attn_weights, in_3):
    attn_weights = attn_weights.contiguous()
    in_3 = in_3.contiguous()

    B, H, N, _ = attn_weights.shape
    D = in_3.shape[-1]

    # Output directly in permuted layout [B, N, H, D]
    out = torch.empty((B, N, H, D), dtype=attn_weights.dtype, device=attn_weights.device)

    N_BLOCK = triton.next_power_of_2(N)
    D_BLOCK = triton.next_power_of_2(D)
    IS_BF16 = (attn_weights.dtype == torch.bfloat16)

    # Grid: (B*H*N,) - i varies fastest for value matrix L2 cache reuse
    grid = (B * H * N,)
    _matmul_permute_kernel[grid](
        attn_weights, in_3, out,
        B, H, N, D,
        N_BLOCK=N_BLOCK,
        D_BLOCK=D_BLOCK,
        IS_BF16=IS_BF16,
    )
    return out


def pattern(attn_weights, in_3):
    matmul = torch.matmul(attn_weights, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(attn_weights, in_3):
    return (attn_weights, in_3)


def replacement_func():
    return _matmul_permute_wrapper