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
def _dropout_matmul_permute_kernel(
    attn_ptr,   # [B, H, N, N] contiguous - attention weights (post-softmax)
    value_ptr,  # [B, H, N, D] contiguous - value
    out_ptr,    # [B, N, H, D] contiguous - output in permuted layout
    B, H, N, D,
    N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # Each program computes one output row: out[b, i, h, :]
    # i varies fastest (same h → value matrix stays in L2 cache)
    pid = tl.program_id(0)
    i = pid % N
    h = (pid // N) % H
    b = pid // (N * H)

    k_range = tl.arange(0, N_BLOCK)
    k_mask = k_range < N

    # Load attention weights row (already post-softmax; dropout is identity at inference)
    # attn[b, h, i, :] with strides [H*N*N, N*N, N, 1]
    attn_base = b * H * N * N + h * N * N + i * N
    attn_w = tl.load(attn_ptr + attn_base + k_range, mask=k_mask, other=0.0).to(tl.float32)

    # Load value matrix: value[b, h, :, :] with strides [H*N*D, N*D, D, 1]
    d_range = tl.arange(0, D_BLOCK)
    d_mask = d_range < D
    v_base = b * H * N * D + h * N * D
    v_2d = tl.load(
        value_ptr + v_base + k_range[:, None] * D + d_range[None, :],
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float32)

    # Weighted sum (dropout is identity with training=False, no scaling needed)
    acc = tl.sum(attn_w[:, None] * v_2d, axis=0)  # [D_BLOCK]

    # Write in permuted [B, N, H, D] layout
    out_base = b * N * H * D + i * H * D + h * D
    if IS_BF16:
        tl.store(out_ptr + out_base + d_range, acc.to(tl.bfloat16), mask=d_mask)
    else:
        tl.store(out_ptr + out_base + d_range, acc.to(tl.float16), mask=d_mask)


@torch.fx.wrap
def _dropout_matmul_permute_wrapper(softmax_out, in_3):
    softmax_out = softmax_out.contiguous()
    in_3 = in_3.contiguous()

    B, H, N, _ = softmax_out.shape
    D = in_3.shape[-1]

    out = torch.empty((B, N, H, D), dtype=softmax_out.dtype, device=softmax_out.device)

    N_BLOCK = triton.next_power_of_2(N)
    D_BLOCK = triton.next_power_of_2(D)
    IS_BF16 = (softmax_out.dtype == torch.bfloat16)

    grid = (B * H * N,)
    _dropout_matmul_permute_kernel[grid](
        softmax_out, in_3, out,
        B, H, N, D,
        N_BLOCK=N_BLOCK,
        D_BLOCK=D_BLOCK,
        IS_BF16=IS_BF16,
    )
    return out


def pattern(softmax_out, in_3):
    # Match: dropout(identity) + matmul + permute + contiguous
    attn_weights = torch.nn.functional.dropout(softmax_out, 0.1, False, False)
    matmul = torch.matmul(attn_weights, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(softmax_out, in_3):
    return (softmax_out, in_3)


def replacement_func():
    return _dropout_matmul_permute_wrapper