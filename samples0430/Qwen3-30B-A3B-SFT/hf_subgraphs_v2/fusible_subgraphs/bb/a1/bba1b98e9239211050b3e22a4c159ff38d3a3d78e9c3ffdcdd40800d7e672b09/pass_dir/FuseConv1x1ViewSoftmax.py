import torch
import triton
import triton.language as tl


def pattern(bias, weight, x):
    conv2d = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    view = conv2d.view(-1, 1, -1)
    softmax = view.softmax(dim=-1)
    return softmax


def replacement_args(bias, weight, x):
    return (bias, weight, x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 4096}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8,  num_stages=3),
    ],
    key=['K', 'HW'],
)
@triton.jit
def conv1x1_softmax_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B, K, HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Each program handles one batch item b.
    Computes: output[b, 0, hw] = softmax(dot(input[b, :, hw], weight[0, :, 0, 0]) + bias[0])
    This fuses 1x1 conv (with 1 output channel) + view (spatial reshape) + softmax.
    """
    pid = tl.program_id(0)
    b = pid

    # Channel stride in NCHW layout: stride along channel dim = HW
    stride_k = HW
    # Stride along batch dim = K * HW
    stride_b = K * HW

    hw_offsets = tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW

    # Load scalar bias and promote to fp32
    bias_val = tl.load(bias_ptr).to(tl.float32)

    # Accumulate dot products over K channels
    # offset = b * K * HW  (batch start)
    # x[b, k, hw] = input_ptr[b * K * HW + k * HW + hw]
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    for k in range(K):
        # weight layout [1, K, 1, 1] -> weight[0, k, 0, 0] = weight_ptr[k]
        w = tl.load(weight_ptr + k).to(tl.float32)
        x = tl.load(
            input_ptr + b * stride_b + k * stride_k + hw_offsets,
            mask=mask, other=0.0
        ).to(tl.float32)
        acc += w * x

    # Add bias
    acc += bias_val

    # Softmax in fp32 for numerical stability
    max_val = tl.max(acc, axis=0)
    acc = tl.exp(acc - max_val)
    sum_val = tl.sum(acc, axis=0)
    acc = acc / sum_val

    # Store back in original dtype
    out = acc.to(x.dtype)
    tl.store(output_ptr + b * HW + hw_offsets, out, mask=mask)


@torch.fx.wrap
def conv1x1_view_softmax(bias, weight, x):
    B, K, H, W = x.shape
    HW = H * W
    # Output shape matches view(conv2d) + softmax = [B, 1, HW]
    output = torch.empty((B, 1, HW), dtype=x.dtype, device=x.device)

    conv1x1_softmax_kernel[(B,)](
        x, weight, bias, output,
        B, K, HW,
    )
    return output


def replacement_func():
    return conv1x1_view_softmax