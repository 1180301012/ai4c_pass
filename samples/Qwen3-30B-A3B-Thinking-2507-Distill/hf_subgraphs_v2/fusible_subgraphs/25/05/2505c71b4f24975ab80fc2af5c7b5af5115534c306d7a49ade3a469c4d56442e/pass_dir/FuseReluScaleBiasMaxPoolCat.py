import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, relu_out):
    # Match scale * relu_out + bias  (mul + add, skipping relu)
    # These use purely positional args — no kwargs — so ForceArgsTracer
    # normalization won't cause args/kwargs length mismatches.
    tmp_3 = in_1 * relu_out
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, relu_out):
    return (in_0, in_1, relu_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['N'],
)
@triton.jit
def fused_scale_add_kernel(
    bias_ptr,    # in_0: bias shape [1]
    scale_ptr,   # in_1: scale shape [1]
    x_ptr,       # relu_out: already relu-activated tensor
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    bias  = tl.load(bias_ptr)
    scale = tl.load(scale_ptr)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # x is already relu output; just scale and add bias
    out = scale * x + bias
    tl.store(out_ptr + offs, out, mask=mask)


@torch.fx.wrap
def fused_relu_scale_bias(in_0, in_1, relu_out):
    N   = relu_out.numel()
    out = torch.empty_like(relu_out)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fused_scale_add_kernel[grid](in_0, in_1, relu_out, out, N)
    return out


def replacement_func():
    return fused_relu_scale_bias