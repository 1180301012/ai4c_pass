import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_total'],
)
@triton.jit
def _fused_relu_dropout2d_kernel(
    in_ptr,
    out0_ptr,
    out1_ptr,
    n_total,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused relu + dropout2d (inference, no-op dropout) kernel.

    - Reads input once
    - Applies ReLU, writing to out0_ptr (relu output, tmp_0)
    - Writes same data to out1_ptr (dropout2d no-op output, tmp_1 = relu output)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_total

    x = tl.load(in_ptr + offsets, mask=mask)
    # ReLU: max(x, 0)
    y = tl.maximum(x, 0)
    # Write relu output (tmp_0)
    tl.store(out0_ptr + offsets, y, mask=mask)
    # Write dropout output (tmp_1 = relu output, dropout is no-op when training=False)
    tl.store(out1_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def fused_relu_dropout2d(in_0):
    n_total = in_0.numel()
    # Both outputs are relu outputs; dropout2d with training=False is identity
    out_relu = torch.empty_like(in_0)
    out_dropout = torch.empty_like(in_0)

    grid = lambda meta: (triton.cdiv(n_total, meta['BLOCK_SIZE']),)

    _fused_relu_dropout2d_kernel[grid](
        in_ptr=in_0,
        out0_ptr=out_relu,
        out1_ptr=out_dropout,
        n_total=n_total,
    )

    # dropout2d(tmp_0, 0.1, False, False) returns tmp_0 when training=False
    # so we return (out_dropout, out_relu) == (relu_out, relu_out)
    return out_dropout, out_relu


def replacement_func():
    return fused_relu_dropout2d