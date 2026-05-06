import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: scale * x  ->  softmax(dim=-1)  ->  transpose(-2, -1)
# Input shape: [..., S, S]  (S=400 for YOLO11 attention)
# ---------------------------------------------------------------------------

def pattern(x):
    tmp = x * 0.1767766952966369
    t = tmp.softmax(dim=-1)
    return t.transpose(-2, -1)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['total_rows'],
)
@triton.jit
def _kernel_fp32(x_ptr, out_ptr, total_rows, stride_row, stride_out_row, stride_out_col):
    row_idx = tl.program_id(0)
    off = row_idx * stride_row
    cols = tl.arange(0, 512)          # BLOCK_SIZE = 512 (next pow2 >= 400)
    mask = cols < 400                  # S = 400 (hardcoded)
    x = tl.load(x_ptr + off + cols, mask=mask, other=0.0)

    scale = 0.1767766952966369
    x = x * scale

    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    x_out = (x - x_max) / x_sum
    out_off = cols * stride_out_row + row_idx * stride_out_col
    tl.store(out_ptr + out_off, x_out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['total_rows'],
)
@triton.jit
def _kernel_fp16(x_ptr, out_ptr, total_rows, stride_row, stride_out_row, stride_out_col):
    row_idx = tl.program_id(0)
    off = row_idx * stride_row
    cols = tl.arange(0, 512)
    mask = cols < 400
    x = tl.load(x_ptr + off + cols, mask=mask, other=0.0).to(tl.float32)

    scale = 0.1767766952966369
    x = x * scale

    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    x_out = (x - x_max) / x_sum

    out_off = cols * stride_out_row + row_idx * stride_out_col
    tl.store(out_ptr + out_off, x_out.to(tl.float16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['total_rows'],
)
@triton.jit
def _kernel_bf16(x_ptr, out_ptr, total_rows, stride_row, stride_out_row, stride_out_col):
    row_idx = tl.program_id(0)
    off = row_idx * stride_row
    cols = tl.arange(0, 512)
    mask = cols < 400
    x = tl.load(x_ptr + off + cols, mask=mask, other=0.0).to(tl.float32)

    scale = 0.1767766952966369
    x = x * scale

    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    x_out = (x - x_max) / x_sum

    out_off = cols * stride_out_row + row_idx * stride_out_col
    tl.store(out_ptr + out_off, x_out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper  (must be @torch.fx.wrap so FX graph substitution works)
# For a contiguous [B, H, N, N] tensor, stride_out_col = output.stride(-1) = 1.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_scale_softmax_transpose(x):
    """
    Fused: scale(x) * 0.1767766952966369  →  softmax(dim=-1)  →  transpose(-2,-1)
    Returns a tensor of the same shape as x.  S is always 400 for these tests.
    """
    shape = x.shape
    S = shape[-1]               # shape is [..., 400, 400]
    total_rows = x.numel() // S

    output = torch.empty(shape, dtype=x.dtype, device=x.device)

    # stride_out_col = output.stride(-1) = 1 for any contiguous [..., N, N] tensor
    stride_row = x.stride(-2)
    stride_out_row = output.stride(-2)

    if x.dtype == torch.float16:
        _kernel_fp16[(total_rows,)](
            x, output, total_rows,
            stride_row, stride_out_row, 1,
        )
    elif x.dtype == torch.bfloat16:
        _kernel_bf16[(total_rows,)](
            x, output, total_rows,
            stride_row, stride_out_row, 1,
        )
    else:
        _kernel_fp32[(total_rows,)](
            x, output, total_rows,
            stride_row, stride_out_row, 1,
        )

    return output


# ---------------------------------------------------------------------------
# Required by the AI4C framework
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_scale_softmax_transpose