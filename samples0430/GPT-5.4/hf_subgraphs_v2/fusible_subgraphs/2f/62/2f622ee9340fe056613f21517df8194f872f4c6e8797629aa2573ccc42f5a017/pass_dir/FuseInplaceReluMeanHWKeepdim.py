import torch
import triton
import triton.language as tl


# Match the observable subgraph exactly: the in-place ReLU result is both returned
# and consumed by the mean reduction.
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)


# The integer scalar path is dead for observable outputs, so the replacement only
# needs the activation tensor.
def replacement_args(in_0, in_1):
    return (in_1,)


@triton.jit
def _relu_mean_inplace_kernel(
    x_ptr,
    mean_ptr,
    C,
    H,
    W,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_mn,
    stride_mc,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base_x = x_ptr + n * stride_xn + c * stride_xc
    hw_offsets = tl.arange(0, BLOCK_HW)
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    hw = H * W

    for block_idx in range(0, tl.cdiv(hw, BLOCK_HW)):
        offs = block_idx * BLOCK_HW + hw_offsets
        mask = offs < hw
        h = offs // W
        w = offs - h * W
        ptrs = base_x + h * stride_xh + w * stride_xw
        vals = tl.load(ptrs, mask=mask, other=0.0)
        relu_vals = tl.maximum(vals, 0.0)
        tl.store(ptrs, relu_vals, mask=mask)
        acc += relu_vals.to(tl.float32)

    mean_val = tl.sum(acc, axis=0) / hw
    mean_ptrs = mean_ptr + n * stride_mn + c * stride_mc
    tl.store(mean_ptrs, mean_val)


@torch.fx.wrap
def fused_inplace_relu_mean_hw_keepdim(x):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]

    mean_out = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)
    if n == 0 or c == 0:
        return x, mean_out

    hw = h * w
    if hw <= 64:
        block_hw = 64
        num_warps = 2
    elif hw <= 256:
        block_hw = 128
        num_warps = 4
    else:
        block_hw = 256
        num_warps = 8

    grid = (n * c,)
    _relu_mean_inplace_kernel[grid](
        x,
        mean_out,
        c,
        h,
        w,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        mean_out.stride(0),
        mean_out.stride(1),
        BLOCK_HW=block_hw,
        num_warps=num_warps,
    )
    return x, mean_out


def replacement_func():
    return fused_inplace_relu_mean_hw_keepdim