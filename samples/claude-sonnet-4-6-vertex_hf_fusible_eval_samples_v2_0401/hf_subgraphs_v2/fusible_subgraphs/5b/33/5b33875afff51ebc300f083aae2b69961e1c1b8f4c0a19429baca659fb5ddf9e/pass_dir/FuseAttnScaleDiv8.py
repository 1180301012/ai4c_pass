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
def _fused_scaled_attn_kernel_div8(
    attn_scores_ptr,  # [B, H, N, N] contiguous
    mask_ptr,         # [1, 1, 1, N] contiguous
    value_ptr,        # [B, H, N, D] contiguous
    out_ptr,          # [B, N, H, D] contiguous (output in permuted layout)
    inv_scale,
    B, H, N, D,
    N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # Each program computes one attention row: out[b, i, h, :] for a fixed (b, h, i)
    pid = tl.program_id(0)
    i = pid % N
    h = (pid // N) % H
    b = pid // (N * H)

    # Index range for the key/sequence dimension
    k_range = tl.arange(0, N_BLOCK)
    k_mask = k_range < N

    # Load attention scores row: attn_scores[b, h, i, :] and scale
    # Strides for [B, H, N, N] contiguous: [H*N*N, N*N, N, 1]
    as_base = b * H * N * N + h * N * N + i * N
    scores = tl.load(
        attn_scores_ptr + as_base + k_range,
        mask=k_mask, other=0.0
    ).to(tl.float32)
    scores = scores * inv_scale

    # Add attention mask: mask[0, 0, 0, :] -- shape [1,1,1,N], stride-1 in last dim
    mask_vals = tl.load(
        mask_ptr + k_range,
        mask=k_mask, other=0.0
    ).to(tl.float32)
    scores = scores + mask_vals

    # Softmax over the key dimension (dim=-1)
    scores = tl.where(k_mask, scores, float('-inf'))
    max_score = tl.max(scores, axis=0)
    exp_s = tl.exp(scores - max_score)
    exp_s = tl.where(k_mask, exp_s, 0.0)
    sum_exp = tl.sum(exp_s, axis=0)
    softmax_w = exp_s / sum_exp  # [N_BLOCK]

    # Load value matrix: value[b, h, :, :] -- [N_BLOCK, D_BLOCK]
    # Strides for [B, H, N, D] contiguous: [H*N*D, N*D, D, 1]
    d_range = tl.arange(0, D_BLOCK)
    d_mask = d_range < D

    v_base = b * H * N * D + h * N * D
    v_2d = tl.load(
        value_ptr + v_base + k_range[:, None] * D + d_range[None, :],
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float32)

    # Weighted sum: out[b, i, h, :] = sum_k softmax_w[k] * value[b, h, k, :]
    acc = tl.sum(softmax_w[:, None] * v_2d, axis=0)  # [D_BLOCK]

    # Store directly in permuted layout: out[b, i, h, :]
    # Strides for [B, N, H, D] contiguous: [N*H*D, H*D, D, 1]
    out_base = b * N * H * D + i * H * D + h * D
    if IS_BF16:
        tl.store(out_ptr + out_base + d_range, acc.to(tl.bfloat16), mask=d_mask)
    else:
        tl.store(out_ptr + out_base + d_range, acc.to(tl.float16), mask=d_mask)


@torch.fx.wrap
def _fused_attn_scale_div8_wrapper(in_0, in_2, in_3):
    # Ensure contiguous inputs for offset arithmetic
    in_0 = in_0.contiguous()
    in_2 = in_2.contiguous()
    in_3 = in_3.contiguous()

    B, H, N, _ = in_0.shape
    D = in_3.shape[-1]

    # Allocate output in the permuted layout [B, N, H, D] directly
    out = torch.empty((B, N, H, D), dtype=in_0.dtype, device=in_0.device)

    inv_scale = 1.0 / 8.0
    # Block sizes must be >= N and D respectively (power of 2)
    N_BLOCK = triton.next_power_of_2(N)
    D_BLOCK = triton.next_power_of_2(D)
    IS_BF16 = (in_0.dtype == torch.bfloat16)

    grid = (B * H * N,)

    _fused_scaled_attn_kernel_div8[grid](
        in_0, in_2, in_3, out,
        inv_scale,
        B, H, N, D,
        N_BLOCK=N_BLOCK,
        D_BLOCK=D_BLOCK,
        IS_BF16=IS_BF16,
    )

    return out


def pattern(in_0, in_2, in_3):
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


def replacement_func():
    return _fused_attn_scale_div8_wrapper