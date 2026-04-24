import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def fused_softmax_weighted_sum_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: softmax(x, dim=1) * [0,1,2,3,4] -> sum(dim=1) -> 5 - result
    Each program handles one row of the input tensor [B, n_cols].
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load the row of the input tensor
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0)

    # Upcast to float32 for numerically stable softmax and arithmetic
    x_f32 = x.to(tl.float32)

    # Numerically stable softmax along the column dimension
    x_max = tl.max(x_f32, axis=0)
    x_shifted = x_f32 - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum

    # Weighted sum with positions [0, 1, 2, 3, 4]
    pos = offsets.to(tl.float32)
    weighted = x_softmax * pos
    weighted_sum = tl.sum(weighted, axis=0)

    # 5 - weighted_sum
    result = 5.0 - weighted_sum

    # Downcast back to the original dtype and store
    result_cast = result.to(x.dtype)
    tl.store(out_ptr + row, result_cast)


# Updated kernel for multiply+sum+subtract (without softmax)
@triton.jit
def fused_weighted_sum_kernel(
    x_ptr,
    pos_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: (x * pos) -> sum(dim=1) -> 5 - result
    x: [B, n_cols], pos: [n_cols] (positions 0..n_cols-1)
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load x values for this row
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0)
    # Load position values (weights)
    pos = tl.load(pos_ptr + offsets, mask=mask, other=0.0)

    # Upcast to float32
    x_f32 = x.to(tl.float32)
    pos_f32 = pos.to(tl.float32)

    # Weighted sum: (x * pos).sum(dim=0)
    weighted = x_f32 * pos_f32
    weighted_sum = tl.sum(weighted, axis=0)

    # 5 - weighted_sum  (compute in float32 to match eager float32 output)
    result = 5.0 - weighted_sum

    # Explicitly store as float32 regardless of input dtype
    tl.store(out_ptr + row, result)


@torch.fx.wrap
def fused_softmax_weighted_sum(tmp_0, tmp_1):
    """
    Wrapper that launches the fused kernel.
    Input:  tmp_0  softmax output [B, n_cols] (bfloat16 or float16)
            tmp_1  linspace tensor [n_cols] = [0,1,2,3,4]
    Output: [B] tensor with result = 5 - sum(tmp_0 * tmp_1, dim=1)
            Always float32 to match eager float32 output.
    """
    B = tmp_0.shape[0]
    n_cols = tmp_0.shape[1]
    BLOCK_SIZE = 8  # power-of-2 >= n_cols (n_cols=5)

    # Output is float32 to match eager float32 semantics of 5 - scalar
    out = torch.empty(B, dtype=torch.float32, device=tmp_0.device)

    fused_weighted_sum_kernel[(B,)](
        tmp_0,
        tmp_1,
        out,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API consumed by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(tmp_0, tmp_1):
    tmp_2 = tmp_0 * tmp_1
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4


def replacement_args(tmp_0, tmp_1):
    return (tmp_0, tmp_1)


def replacement_func():
    return fused_softmax_weighted_sum