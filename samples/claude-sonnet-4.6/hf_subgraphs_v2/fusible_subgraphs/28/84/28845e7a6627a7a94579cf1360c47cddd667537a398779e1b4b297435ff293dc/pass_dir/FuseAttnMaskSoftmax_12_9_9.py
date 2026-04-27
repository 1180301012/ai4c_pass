import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _attn_softmax_kernel_12_9_9(
    in0_ptr, in1_ptr, out_ptr,
    in0_s2, in0_s3,
    in1_s1, in1_s2, in1_s3,
    N, H, W,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // H
    h = pid % H

    offsets = tl.arange(0, BLOCK_W)
    mask = offsets < W

    # in0: shape [1, 1, H, W]  -> in0[0, 0, h, :]
    in0_base = h * in0_s2 + offsets * in0_s3
    # in1: shape [1, N, H, W]  -> in1[0, n, h, :]
    in1_base = n * in1_s1 + h * in1_s2 + offsets * in1_s3

    x0 = tl.load(in0_ptr + in0_base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in1_ptr + in1_base, mask=mask, other=0.0).to(tl.float32)

    x = x0 + x1
    # torch.max with float32 constant -3.4028234663852886e+38
    x = tl.maximum(x, -3.4028234663852886e+38)

    # Safe softmax along W (last) dimension
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum

    # Output: contiguous [N, H, W] in float32 (dropout training=False is no-op)
    out_base = (n * H + h) * W
    tl.store(out_ptr + out_base + offsets, x_softmax, mask=mask)


@torch.fx.wrap
def triton_attn_softmax_12_9_9(in_0, in_1):
    N, H, W = 12, 9, 9
    out = torch.empty((N, H, W), dtype=torch.float32, device=in_1.device)

    in0_s2 = in_0.stride(2)
    in0_s3 = in_0.stride(3)
    in1_s1 = in_1.stride(1)
    in1_s2 = in_1.stride(2)
    in1_s3 = in_1.stride(3)

    grid = (N * H,)
    _attn_softmax_kernel_12_9_9[grid](
        in_0, in_1, out,
        in0_s2, in0_s3,
        in1_s1, in1_s2, in1_s3,
        N, H, W,
        BLOCK_W=16,
    )
    return out


def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(12, 9, 9)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_attn_softmax_12_9_9