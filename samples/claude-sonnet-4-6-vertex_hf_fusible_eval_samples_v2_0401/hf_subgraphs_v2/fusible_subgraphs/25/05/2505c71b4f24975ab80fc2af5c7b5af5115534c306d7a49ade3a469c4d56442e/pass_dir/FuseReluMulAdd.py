import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fuse relu + mul + add:  relu(x) * y + z
# In the model: relu(in_2) * in_1 + in_0
#   x = in_2  (activation, shape [B,C,H,W])
#   y = in_1  (scale, shape [1])
#   z = in_0  (bias,  shape [1])
#
# Uses aten.relu.default since dynamo lowers F.relu to aten.
# operator.mul/add kept as Python operators (confirmed matched by FuseMulAdd).
# ---------------------------------------------------------------------------
def pattern(x, y, z):
    relu_x = torch.nn.functional.relu(x, inplace=False)
    tmp    = y * relu_x
    return tmp + z


def replacement_args(x, y, z):
    return (x, y, z)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=4),
    ],
    key=['N'],
)
@triton.jit
def triton_relu_scale_bias_kernel(
    x_ptr,    # activation – shape [B,C,H,W]
    y_ptr,    # scale      – shape [1]
    z_ptr,    # bias       – shape [1]
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    scale = tl.load(y_ptr)   # broadcast scalar
    bias  = tl.load(z_ptr)   # broadcast scalar
    x     = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    relu_x = tl.maximum(x, 0.0)
    out    = scale * relu_x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_relu_scale_bias(x, y, z):
    # x = activation [B,C,H,W], y = scale [1], z = bias [1]
    N   = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    triton_relu_scale_bias_kernel[grid](x, y, z, out, N)
    return out


def replacement_func():
    return triton_relu_scale_bias