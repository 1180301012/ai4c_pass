import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 512}, num_warps=16),
        triton.Config({'BLOCK_M': 512}, num_warps=32),
        triton.Config({'BLOCK_M': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 1024}, num_warps=16),
        triton.Config({'BLOCK_M': 1024}, num_warps=32),
    ],
    key=['N_COLS'],
)
@triton.jit
def fused_max_softmax_kernel(
    input_ptr,
    output_ptr,
    N_ROWS,
    N_COLS,
    BLOCK_M: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_M)
    mask = col_offsets < N_COLS

    # Load input row; upcast to fp32 for numerical stability
    x = tl.load(
        input_ptr + row_idx * N_COLS + col_offsets,
        mask=mask,
        other=float('-inf'),
    ).to(tl.float32)

    # Step 1: find max of the row  (mirrors torch.max(..., keepdim=True))
    max_x = tl.max(x, axis=0)

    # Step 2: shifted = max_x - x  (mirrors expand_as + subtract)
    # Mask out-of-bounds lanes to -inf so they contribute 0 to the sum
    shifted = tl.where(mask, max_x - x, float('-inf'))

    # Step 3: numerically-stable softmax of shifted
    max_shifted = tl.max(shifted, axis=0)       # max(shifted) = max_x - min(x)
    stable = shifted - max_shifted               # ≤ 0 for all valid lanes
    exp_val = tl.exp(stable)
    exp_val = tl.where(mask, exp_val, 0.0)
    sum_exp = tl.sum(exp_val, axis=0)
    out = exp_val / sum_exp

    # Store result (Triton automatically down-casts fp32→fp16/bf16 if needed)
    tl.store(
        output_ptr + row_idx * N_COLS + col_offsets,
        out,
        mask=mask,
    )


@torch.fx.wrap
def fused_max_softmax(in_0):
    orig_shape = in_0.shape
    N_ROWS = in_0.numel() // orig_shape[-1]
    N_COLS = orig_shape[-1]

    x_cont = in_0.contiguous().view(N_ROWS, N_COLS)
    out = torch.empty_like(x_cont)

    fused_max_softmax_kernel[(N_ROWS,)](
        x_cont,
        out,
        N_ROWS,
        N_COLS,
    )

    return out.view(orig_shape)


def replacement_func():
    return fused_max_softmax