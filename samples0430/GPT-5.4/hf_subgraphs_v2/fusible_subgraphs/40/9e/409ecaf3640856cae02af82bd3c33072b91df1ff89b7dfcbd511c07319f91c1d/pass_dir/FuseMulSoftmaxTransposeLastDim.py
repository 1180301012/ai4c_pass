import torch
import triton
import triton.language as tl


_SCALE = 0.1767766952966369


# Pattern matching function
def pattern(in_0: torch.Tensor):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2


# Argument extraction function
def replacement_args(in_0: torch.Tensor):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 1}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 1}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 2}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512, "ROWS_PER_PROGRAM": 4}, num_warps=8, num_stages=2),
    ],
    key=["n_cols", "num_rows"],
)
@triton.jit
def _fused_scaled_softmax_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    n_cols,
    scale,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ids = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    ptrs = x_ptr + row_ids[:, None] * n_cols + col_offsets[None, :]
    mask = (row_ids[:, None] < num_rows) & (col_offsets[None, :] < n_cols)

    x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x * scale

    row_max = tl.max(x, axis=1)[:, None]
    z = (x - row_max) * 1.4426950408889634
    num = tl.exp2(z)
    den = tl.sum(num, axis=1)[:, None]
    y = num / den

    out_ptrs = out_ptr + row_ids[:, None] * n_cols + col_offsets[None, :]
    tl.store(out_ptrs, y, mask=mask)


# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_scaled_softmax_transpose_lastdim(in_0: torch.Tensor):
    n_cols = in_0.shape[-1]
    num_rows = in_0.numel() // n_cols

    out = torch.empty(in_0.shape, device=in_0.device, dtype=in_0.dtype)

    grid = lambda meta: (triton.cdiv(num_rows, meta["ROWS_PER_PROGRAM"]),)
    _fused_scaled_softmax_kernel[grid](
        in_0,
        out,
        num_rows,
        n_cols,
        _SCALE,
    )

    return out.transpose(-2, -1)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_scaled_softmax_transpose_lastdim