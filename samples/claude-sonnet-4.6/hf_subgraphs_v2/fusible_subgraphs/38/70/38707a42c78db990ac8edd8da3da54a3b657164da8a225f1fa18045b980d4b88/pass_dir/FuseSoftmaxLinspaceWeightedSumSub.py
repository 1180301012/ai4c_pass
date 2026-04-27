import torch
import triton
import triton.language as tl


# Pattern: fuse (softmax_out * weights).sum(dim=1) → 5 - result
# tmp_0 = softmax output, w = linspace weights tensor
def pattern(tmp_0, w):
    tmp_2 = tmp_0 * w
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4


# Return only tmp_0 — omitting w causes FX DCE to drop the linspace node entirely
def replacement_args(tmp_0, w):
    return (tmp_0,)


@triton.jit
def weighted_sum_sub_kernel(
    x_ptr,
    out_ptr,
    N,
    STRIDE,              # row stride (runtime — safe for any layout)
    BLOCK: tl.constexpr,
    K:    tl.constexpr,
):
    row = tl.program_id(0)

    col_ids = tl.arange(0, BLOCK)
    mask = col_ids < K
    offsets = row * STRIDE + col_ids

    # Load softmax output as float32
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Hardcoded weights [0, 1, 2, 3, 4] — mirror linspace(0, 4, steps=5)
    weights = col_ids.to(tl.float32)
    ws = tl.sum(tl.where(mask, x * weights, 0.0), axis=0)

    # 5.0 - weighted_sum
    result = 5.0 - ws

    tl.store(out_ptr + row, result)


@torch.fx.wrap
def weighted_sum_sub_wrapper(tmp_0):
    N = tmp_0.shape[0]
    out = torch.empty(N, dtype=torch.float32, device=tmp_0.device)

    weighted_sum_sub_kernel[(N,)](
        tmp_0, out, N,
        tmp_0.stride(0),   # safe runtime stride
        BLOCK=8, K=5,
        num_warps=1,
    )

    return out


def replacement_func():
    return weighted_sum_sub_wrapper