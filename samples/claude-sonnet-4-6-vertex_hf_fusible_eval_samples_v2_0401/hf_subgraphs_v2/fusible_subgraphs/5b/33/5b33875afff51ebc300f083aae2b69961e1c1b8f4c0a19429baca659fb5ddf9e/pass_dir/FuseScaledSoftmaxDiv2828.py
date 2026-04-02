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
    key=['B', 'H', 'N'],
)
@triton.jit
def _scaled_softmax_div2828_kernel(
    scores_ptr,  # [B, H, N, N] contiguous
    mask_ptr,    # [1, 1, 1, N] contiguous
    out_ptr,     # [B, H, N, N] contiguous output
    inv_scale,
    B, H, N,
    N_BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid % N
    h = (pid // N) % H
    b = pid // (N * H)

    k_range = tl.arange(0, N_BLOCK)
    k_mask = k_range < N

    row_base = b * H * N * N + h * N * N + i * N
    scores = tl.load(scores_ptr + row_base + k_range, mask=k_mask, other=0.0).to(tl.float32)
    scores = scores * inv_scale

    mask_vals = tl.load(mask_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)
    scores = scores + mask_vals

    scores = tl.where(k_mask, scores, float('-inf'))
    max_s = tl.max(scores, axis=0)
    exp_s = tl.exp(scores - max_s)
    exp_s = tl.where(k_mask, exp_s, 0.0)
    sum_exp = tl.sum(exp_s, axis=0)
    softmax_out = exp_s / sum_exp

    if IS_BF16:
        tl.store(out_ptr + row_base + k_range, softmax_out.to(tl.bfloat16), mask=k_mask)
    else:
        tl.store(out_ptr + row_base + k_range, softmax_out.to(tl.float16), mask=k_mask)


@torch.fx.wrap
def _scaled_softmax_div2828_wrapper(in_0, in_2):
    in_0 = in_0.contiguous()
    in_2 = in_2.contiguous()

    B, H, N, _ = in_0.shape
    out = torch.empty_like(in_0)

    inv_scale = 1.0 / 2.8284271247461903
    N_BLOCK = triton.next_power_of_2(N)
    IS_BF16 = (in_0.dtype == torch.bfloat16)

    grid = (B * H * N,)
    _scaled_softmax_div2828_kernel[grid](
        in_0, in_2, out,
        inv_scale,
        B, H, N,
        N_BLOCK=N_BLOCK,
        IS_BF16=IS_BF16,
    )
    return out


def pattern(in_0, in_2):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    return tmp_2


def replacement_args(in_0, in_2):
    return (in_0, in_2)


def replacement_func():
    return _scaled_softmax_div2828_wrapper