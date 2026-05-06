import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.cumsum(in_0, dim=1)
    tmp_2 = tmp_1 * in_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['num_cols'],
)
@triton.jit
def fused_attn_mask_bias_adaptive_kernel(
    in_ptr,
    out_ptr,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_offset = row_idx * num_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < num_cols

    # Load input row into registers (scalar loads to accumulate)
    x = tl.load(in_ptr + row_offset + cols, mask=mask, other=0)

    # Compute cumulative sum: acc[col] = sum_{j <= col} x[j]
    # Process in chunks of BLOCK_SIZE for correctness when using small BLOCK_SIZE
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    total_offset = row_offset
    while total_offset < row_offset + num_cols:
        x_block = tl.load(in_ptr + total_offset + cols, mask=mask, other=0)
        # acc[col] = acc[col] + x_block[col]  (partial sum accumulation)
        acc += x_block
        total_offset += BLOCK_SIZE

    # Compute fused result: (acc * x - 1) + 2  =  acc * x + 1
    result = acc * x + 1

    # Store output (int64)
    tl.store(out_ptr + row_offset + cols, result, mask=mask)


@torch.fx.wrap
def fused_attn_mask_bias_adaptive(in_0):
    B = in_0.shape[0]
    C = in_0.shape[1]
    N_ROWS = B  # one row per batch element
    num_cols = C

    out = torch.empty_like(in_0)

    def grid(meta):
        return (N_ROWS,)

    fused_attn_mask_bias_adaptive_kernel[grid](
        in_0, out, num_cols,
    )

    return out


def replacement_func():
    return fused_attn_mask_bias_adaptive