import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    in_1 += in_0
    in_2 = in_1
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row_idx * n_cols

    # Load with -inf padding so masked positions don't affect max/sum
    in_0 = tl.load(in_0_ptr + row_start + offsets, mask=mask, other=float('-inf'))
    in_1 = tl.load(in_1_ptr + row_start + offsets, mask=mask, other=float('-inf'))

    # Add in original dtype, then upcast to float32 (mirrors original model)
    x = (in_0 + in_1).to(tl.float32)

    # Numerically-stable softmax:
    # Masked positions have -inf → exp(-inf) = 0, so they don't affect the sum
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum

    # Cast back to original input dtype and store
    out = x_softmax.to(in_0.dtype)
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    n_cols = in_0.shape[-1]
    n_rows = in_0.numel() // n_cols

    # BLOCK_SIZE must be a power of 2 >= n_cols
    BLOCK_SIZE = 32
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2

    # Scale num_warps with block size
    num_warps = max(1, BLOCK_SIZE // 32)

    out = torch.empty_like(in_0)

    fused_add_softmax_kernel[(n_rows,)](
        in_0, in_1, out,
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out


def replacement_func():
    return fused_add_softmax