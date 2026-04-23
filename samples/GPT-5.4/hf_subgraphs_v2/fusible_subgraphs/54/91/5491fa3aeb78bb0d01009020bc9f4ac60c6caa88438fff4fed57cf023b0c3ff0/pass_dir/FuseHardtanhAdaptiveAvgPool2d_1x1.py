import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_hardtanh_gap_nchw_kernel(
    x_ptr,
    out_ptr,
    rows,
    hw,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_ids = tl.arange(0, BLOCK_HW)
    mask = (row_ids[:, None] < rows) & (col_ids[None, :] < hw)
    ptrs = x_ptr + row_ids[:, None] * hw + col_ids[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(tl.minimum(x, 6.0), 0.0)
    y = tl.sum(x, axis=1) / hw
    tl.store(out_ptr + row_ids, y, mask=row_ids < rows)


@torch.fx.wrap
def fused_hardtanh_gap_nchw(in_0):
    n = in_0.shape[0]
    c = in_0.shape[1]
    h = in_0.shape[2]
    w = in_0.shape[3]
    rows = n * c
    hw = h * w
    out = torch.empty((n, c, 1, 1), device=in_0.device, dtype=in_0.dtype)
    if hw <= 16:
        block_rows = 32
        block_hw = 16
        num_warps = 2
    elif hw <= 64:
        block_rows = 16
        block_hw = 64
        num_warps = 4
    elif hw <= 128:
        block_rows = 8
        block_hw = 128
        num_warps = 4
    elif hw <= 256:
        block_rows = 4
        block_hw = 256
        num_warps = 4
    else:
        block_rows = 4
        block_hw = 512
        num_warps = 8
    grid = (triton.cdiv(rows, block_rows),)
    fused_hardtanh_gap_nchw_kernel[grid](
        in_0,
        out,
        rows,
        hw,
        BLOCK_ROWS=block_rows,
        BLOCK_HW=block_hw,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def replacement_func():
    return fused_hardtanh_gap_nchw