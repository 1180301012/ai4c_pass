import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fuse the mul+add chain: x * y + z
# In the model: in_1 * relu_out + in_0
#   x = in_1  (scale,  shape [1])
#   y = relu_out (shape [B,C,H,W])
#   z = in_0  (bias,   shape [1])
# Both Python operators (operator.mul, operator.add) confirmed in model graph.
# ---------------------------------------------------------------------------
def pattern(x, y, z):
    tmp = x * y
    return tmp + z


def replacement_args(x, y, z):
    return (x, y, z)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},   num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=4, num_stages=4),
    ],
    key=['N'],
)
@triton.jit
def triton_scale_bias_kernel(
    x_ptr,   # scale – shape [1]
    y_ptr,   # activation – shape [B,C,H,W]
    z_ptr,   # bias  – shape [1]
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    scale = tl.load(x_ptr)        # broadcast scalar
    bias  = tl.load(z_ptr)        # broadcast scalar
    y     = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    out = scale * y + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_scale_bias(x, y, z):
    # x = scale [1], y = activation [B,C,H,W], z = bias [1]
    # For small N, PyTorch eager ops are faster due to lower kernel launch overhead.
    # Threshold ~2M elements (crossover observed between B=1 and B=32 cases).
    N = y.numel()
    if N < 2_000_000:
        # Fall back to PyTorch for small inputs (avoids Triton overhead)
        return x * y + z
    out = torch.empty_like(y)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    triton_scale_bias_kernel[grid](x, y, z, out, N)
    return out


def replacement_func():
    return triton_scale_bias