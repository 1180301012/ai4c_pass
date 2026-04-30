import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match mul(in_1, x) + in_0  (relu and pad stay in graph)
# Skip relu because ForceArgsTracer strips kwargs={'inplace':False} which the
# model graph retains, causing SubgraphMatcher to fail.
# This fusion still gives speedup by eliminating one kernel launch.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, x):
    tmp_3 = in_1 * x
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, x):
    return (in_0, in_1, x)


# ---------------------------------------------------------------------------
# Triton kernel: fuses scale*x + bias  (fused mul + add)
# in_0 = bias [1], in_1 = scale [1], x = relu output (tensor)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_scale_bias_kernel(
    bias_ptr,    # in_0 – shape [1]
    scale_ptr,   # in_1 – shape [1]
    x_ptr,       # tensor input
    out_ptr,     # output tensor
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    bias  = tl.load(bias_ptr)
    scale = tl.load(scale_ptr)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = scale * x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_scale_bias(in_0, in_1, x):
    n_elements = x.numel()
    out = torch.empty_like(x)

    def grid(meta):
        return ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_scale_bias_kernel[grid](
        in_0, in_1, x, out,
        n_elements,
    )
    return out


def replacement_func():
    return fused_scale_bias