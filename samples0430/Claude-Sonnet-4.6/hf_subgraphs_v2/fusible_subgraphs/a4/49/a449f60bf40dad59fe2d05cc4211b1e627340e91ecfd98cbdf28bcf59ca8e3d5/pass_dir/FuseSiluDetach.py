import torch
import triton
import triton.language as tl


# -------------------------------------------------------------------
# Pattern: matches the exact computation in model.py
# -------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)


# -------------------------------------------------------------------
# Argument extraction
# -------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# -------------------------------------------------------------------
# Optimized Triton SiLU kernel
# SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
# -------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute SiLU in float32 for numerical accuracy, then cast back
    x_f32 = x.to(tl.float32)
    sigmoid_x = tl.sigmoid(x_f32)
    result = (x_f32 * sigmoid_x).to(x.dtype)

    tl.store(out_ptr + offsets, result, mask=mask)


# -------------------------------------------------------------------
# Kernel wrapper - computes silu and passes through in_1, in_2
# detach() is a no-op in inference; we return the same tensors
# -------------------------------------------------------------------
@torch.fx.wrap
def _silu_and_passthrough(in_0, in_1, in_2):
    n = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _silu_kernel[grid](in_0, out, n)

    # tmp_3 = out.detach() is a view of out — same data, return out twice
    return (in_1, in_2, out, out)


# -------------------------------------------------------------------
# Replacement function (zero-arg, returns callable)
# -------------------------------------------------------------------
def replacement_func():
    return _silu_and_passthrough