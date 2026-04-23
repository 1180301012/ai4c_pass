import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.tensor(256, dtype=torch.float32, device=torch.device(type='cuda', index=0))
    tmp_1 = torch.tensor(0.5, device=torch.device(type='cuda', index=0))
    tmp_2 = tmp_0 ** tmp_1
    in_0 /= tmp_2
    tmp_3 = in_0
    tmp_4 = torch.tensor(0.05, device=torch.device(type='cuda', index=0))
    tmp_3 /= tmp_4
    tmp_5 = tmp_3
    tmp_6 = tmp_5.softmax(dim=-1)
    return (tmp_6,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def scaled_softmax_kernel(
    input_ptr,
    output_ptr,
    stride_batch,
    stride_row,
    n_cols,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # Phase 1: Find max in row for numerical stability
    row_max = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(input_ptr + row_idx * stride_row + offsets, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, vals)

    # Phase 2: Compute exp(sum)
    row_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(input_ptr + row_idx * stride_row + offsets, mask=mask, other=-float('inf'))
        exp_vals = tl.exp((vals - row_max) * scale)
        row_sum += tl.sum(exp_vals, axis=0, mask=mask)

    # Phase 3: Normalize and store
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(input_ptr + row_idx * stride_row + offsets, mask=mask, other=-float('inf'))
        exp_vals = tl.exp((vals - row_max) * scale)
        out = exp_vals / row_sum
        tl.store(output_ptr + row_idx * stride_row + offsets, out, mask=mask)


@torch.fx.wrap
def scaled_softmax_fn(in_0):
    # scale = 1 / (sqrt(256) * 0.05) = 1 / 0.8 = 1.25
    scale = 1.25
    B, N, M = in_0.shape
    n_rows = B * N
    n_cols = M

    output = torch.empty_like(in_0)

    BLOCK_SIZE = 1024

    scaled_softmax_kernel[(n_rows,)](
        input_ptr=in_0,
        output_ptr=output,
        stride_batch=in_0.stride(0),
        stride_row=in_0.stride(1),
        n_cols=n_cols,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return scaled_softmax_fn