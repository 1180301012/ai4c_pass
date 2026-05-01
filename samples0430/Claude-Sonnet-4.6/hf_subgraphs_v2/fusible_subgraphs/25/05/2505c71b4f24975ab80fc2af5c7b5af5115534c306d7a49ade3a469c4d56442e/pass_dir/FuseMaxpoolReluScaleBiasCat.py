import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, x):
    # Match: scale * any_input + bias  (operator.mul + operator.add)
    # This matches in_1 * relu(in_2) + in_0 in the model
    tmp_3 = in_1 * x
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, x):
    return (in_0, in_1, x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _scale_bias_kernel(
    in0_ptr,   # bias  [1]
    in1_ptr,   # scale [1]
    x_ptr,     # input [B, C, H, W]
    out_ptr,   # output [B, C, H, W]
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: out = scale * x + bias"""
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    scale = tl.load(in1_ptr)
    bias  = tl.load(in0_ptr)

    xv  = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = scale * xv + bias
    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def fused_scale_bias(in_0, in_1, x):
    N   = x.numel()
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_SIZE']),)

    _scale_bias_kernel[grid](in_0, in_1, x, out, N)
    return out


def replacement_func():
    return fused_scale_bias