import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: element-wise add followed by dropout2d(training=False)
# Since training=False, dropout2d is a pure identity pass-through.
# We fuse both into a single memory-efficient Triton add kernel.
# ---------------------------------------------------------------------------

def pattern(x, y):
    tmp_3 = x + y
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4


def replacement_args(x, y):
    return (x, y)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_add_identity_kernel(
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


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_dropout2d(x, y):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _fused_add_identity_kernel[grid](x, y, out, n_elements)
    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument factory returning the callable
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_add_dropout2d