import torch
import triton
import triton.language as tl


def pattern(x, y):
    """
    Match: element-wise add followed by dropout2d with training=False (identity).
    Mirrors: tmp_3 = in_4 + in_3
             tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    """
    tmp = x + y
    out = torch.nn.functional.dropout2d(tmp, 0.1, False, False)
    return out


def replacement_args(x, y):
    return (x, y)


# ---------------------------------------------------------------------------
# Triton kernel: element-wise addition (dropout with training=False is identity)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@torch.fx.wrap
def fused_add_dropout2d(x, y):
    """
    Fused replacement for (x + y) followed by dropout2d with training=False.
    Since dropout2d(training=False) is identity, this is just x + y.
    """
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    _fused_add_kernel[grid](x, y, out, N)
    return out


def replacement_func():
    return fused_add_dropout2d