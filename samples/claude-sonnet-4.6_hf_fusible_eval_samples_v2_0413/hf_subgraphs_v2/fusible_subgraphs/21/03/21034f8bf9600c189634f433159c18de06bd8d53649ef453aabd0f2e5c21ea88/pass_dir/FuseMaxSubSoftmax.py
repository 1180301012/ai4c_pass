import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['cols'],
)
@triton.jit
def fused_max_sub_softmax_kernel(
    input_ptr,
    output_ptr,
    total_rows,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: softmax(max(x, dim=-1, keepdim=True) - x, dim=-1)

    Mathematically equivalent to softmax(-x), computed stably as:
        exp(min_x - x) / sum(exp(min_x - x_j))
    where min_x = min(x over the row).
    All exp arguments are <= 0, so no overflow is possible.
    """
    pid = tl.program_id(0)
    row_start = pid * cols

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols

    # Load one full row; out-of-bounds positions get -inf so they don't
    # affect the min calculation.
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=float('-inf'))

    # Up-cast to fp32 for numerically stable reductions (matches PyTorch
    # softmax which also computes in fp32 for fp16/bf16 inputs).
    x_fp32 = x.to(tl.float32)

    # Compute min(x) over valid elements using -max(-x) trick.
    neg_x = tl.where(mask, -x_fp32, float('-inf'))
    min_x = -tl.max(neg_x, axis=0)

    # exp(min_x - x_i): arguments are always <= 0  →  values in (0, 1]
    exp_val = tl.where(mask, tl.exp(min_x - x_fp32), 0.0)

    sum_exp = tl.sum(exp_val, axis=0)
    out_fp32 = exp_val / sum_exp

    # Cast back to original dtype and store
    out = out_fp32.to(x.dtype)
    tl.store(output_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def triton_fused_max_sub_softmax(in_0):
    B, rows, cols = in_0.shape
    total_rows = B * rows

    output = torch.empty_like(in_0)

    grid = (total_rows,)
    fused_max_sub_softmax_kernel[grid](
        in_0,
        output,
        total_rows,
        cols,
    )

    return output


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C framework
# ---------------------------------------------------------------------------

def pattern(in_0):
    # Dynamo traces F.softmax through to the method-level x.softmax(dim)
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = tmp_3.softmax(-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return triton_fused_max_sub_softmax