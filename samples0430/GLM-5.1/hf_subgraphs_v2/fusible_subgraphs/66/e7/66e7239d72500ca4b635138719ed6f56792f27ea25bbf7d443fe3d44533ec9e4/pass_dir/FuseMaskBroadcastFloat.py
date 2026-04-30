import torch
import triton
import triton.language as tl


def pattern(in_0, tmp_4):
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    return tmp_7


def replacement_args(in_0, tmp_4):
    return (in_0, tmp_4)


@triton.jit
def mask_broadcast_kernel(
    mask_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # Load mask value for this row (int64 -> float32)
    mask_val = tl.load(mask_ptr + row_idx).to(tl.float32)

    # Store broadcast mask value (float32)
    tl.store(output_ptr + row_start + col_offsets, mask_val, mask=col_mask)


@torch.fx.wrap
def triton_mask_broadcast(in_0, tmp_4):
    n_rows = tmp_4.shape[0] * tmp_4.shape[1]
    n_cols = tmp_4.shape[2]

    output = torch.empty(tmp_4.shape, dtype=torch.float32, device=tmp_4.device)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    mask_broadcast_kernel[grid](
        in_0, output,
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return triton_mask_broadcast