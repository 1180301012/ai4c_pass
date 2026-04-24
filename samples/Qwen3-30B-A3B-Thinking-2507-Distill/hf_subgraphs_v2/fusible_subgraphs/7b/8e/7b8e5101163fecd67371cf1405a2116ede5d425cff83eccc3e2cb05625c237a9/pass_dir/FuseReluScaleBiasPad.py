import torch
import triton
import triton.language as tl
import inspect


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_scale_bias_pad_kernel(
    in2_ptr,
    in1_ptr,
    in0_ptr,
    out_ptr,
    N, C, H, W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    CHW = C * H * W
    HW = H * W

    n = offsets // CHW
    rem = offsets % CHW
    c = rem // HW
    rem2 = rem % HW
    h = rem2 // (W + 1)
    w = rem2 % (W + 1)

    valid = (h < H) & (w < W)

    in2_val = tl.load(in2_ptr + n * CHW + c * HW + h * W + w,
                      mask=mask & valid, other=0.0)
    in1_val = tl.load(in1_ptr, mask=mask & valid, other=1.0)
    in0_val = tl.load(in0_ptr, mask=mask & valid, other=0.0)

    relu_val = tl.where(in2_val > 0.0, in2_val, 0.0)
    result = in1_val * relu_val + in0_val

    tl.store(out_ptr + offsets, result, mask=mask & valid)
    tl.store(out_ptr + offsets, 0.0, mask=mask & ~valid)


@triton.jit
def fused_scale_bias_kernel(
    in2_ptr,
    in1_ptr,
    in0_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load main tensor elements (masked)
    in2_val = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    # in1 and in0 are shape-[1] scalars - load once without mask
    in1_val = tl.load(in1_ptr)
    in0_val = tl.load(in0_ptr)
    result = in1_val * in2_val + in0_val
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_scale_bias(in_0, in_1, in_2):
    N, C, H, W = in_2.shape
    out = torch.empty((N, C, H, W), dtype=in_2.dtype, device=in_2.device)
    n_elements = N * C * H * W
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_scale_bias_kernel[grid](
        in_2, in_1, in_0, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@torch.fx.wrap
def fused_relu_scale_bias_pad(in_0, in_1, in_2):
    N, C, H, W = in_2.shape
    out = torch.empty((N, C, H + 1, W + 1), dtype=in_2.dtype, device=in_2.device)
    n_elements = N * C * (H + 1) * (W + 1)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_relu_scale_bias_pad_kernel[grid](
        in_2, in_1, in_0, out,
        N, C, H, W, n_elements,
    )
    return (out,)


# Pattern: fuse mul + add (scale + bias). No relu in pattern to avoid
# ForceArgsTracer kwarg normalization issues with F.relu.
# relu(in_2) feeds into this as in_2; pad follows in the graph.
def pattern(in_0, in_1, in_2):
    tmp_3 = in_1 * in_2
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_scale_bias