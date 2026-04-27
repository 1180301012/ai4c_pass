"""
Shared dispatch kernels for BiSeNetV2 BGA fusion passes.
Imported by FuseSigmoidMulAdd.py and FuseSigmoidMul.py so that both passes
return the SAME replacement_func(), satisfying the framework's
output_pass_replacement_func_limit check.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: out[i] = y[i] * sigmoid(x[i]) + z[i]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},   num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _sigmoid_mul_add_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out = y * tl.sigmoid(x) + z
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Kernel B: out[i] = y[i] * sigmoid(x[i])
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},   num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _sigmoid_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    out = y * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Single dispatch wrapper shared by all fusion passes.
#
# route="sigmoid_mul_add" : out = b * sigmoid(a) + c    (3 inputs)
# route="sigmoid_mul"     : out = b * sigmoid(a)        (c = dummy b, ignored)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_fn(a, b, c, route):
    N   = a.numel()
    out = torch.empty_like(b)

    if route == "sigmoid_mul_add":
        grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        _sigmoid_mul_add_kernel[grid](a, b, c, out, N)
    else:
        # route == "sigmoid_mul"
        grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        _sigmoid_mul_kernel[grid](a, b, out, N)

    return out