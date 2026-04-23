import torch
import triton
import triton.language as tl

aten = torch._ops.ops.aten


def pattern(in_1):
    tmp_0 = aten.relu.default(in_1, True)
    tmp_3 = aten.mean.dim(tmp_0, [2, 3], True)
    return (tmp_0, tmp_3)


def replacement_args(in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_relu_mean_kernel(
    input_ptr,
    relu_output_ptr,
    mean_output_ptr,
    N,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    acc = 0.0

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        offsets = n * C * HW + c * HW + hw_offsets

        x = tl.load(input_ptr + offsets, hw_mask, other=0.0)
        relu_x = tl.maximum(x, 0.0)
        tl.store(relu_output_ptr + offsets, relu_x, hw_mask)
        # Accumulate in float32 for numerical stability
        # Out-of-bounds elements are 0 (from load other=0.0 and relu(0)=0),
        # so they contribute 0 to the sum
        acc += tl.sum(relu_x.to(tl.float32))

    mean_val = acc / HW
    mean_offset = n * C + c
    tl.store(mean_output_ptr + mean_offset, mean_val)


@torch.fx.wrap
def fused_relu_mean(in_1):
    N, C, H, W = in_1.shape
    HW = H * W

    relu_output = torch.empty_like(in_1)
    mean_output = torch.empty(N, C, 1, 1, dtype=in_1.dtype, device=in_1.device)

    grid = (N * C,)

    fused_relu_mean_kernel[grid](
        input_ptr=in_1,
        relu_output_ptr=relu_output,
        mean_output_ptr=mean_output,
        N=N,
        C=C,
        HW=HW,
    )

    return (relu_output, mean_output)


def replacement_func():
    return fused_relu_mean