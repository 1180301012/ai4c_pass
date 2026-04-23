import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return (tmp_3,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_mul_sum_sigmoid_kernel_lowp(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    base_in = pid_n * 64 * 4096 + offs_hw[None, :]

    for c0 in tl.static_range(0, 64, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        offs = base_in + offs_c[:, None] * 4096
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        acc += tl.sum(x * y, axis=0).to(tl.float32)

    tl.store(out_ptr + pid_n * 4096 + offs_hw, tl.sigmoid(acc))


@triton.jit
def fused_mul_sum_sigmoid_kernel_f32(
    x_ptr,
    y_ptr,
    out_ptr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    base_in = pid_n * 64 * 4096 + offs_hw[None, :]

    for c0 in tl.static_range(0, 64, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        offs = base_in + offs_c[:, None] * 4096
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        acc += tl.sum(x * y, axis=0)

    tl.store(out_ptr + pid_n * 4096 + offs_hw, tl.sigmoid(acc))


@torch.fx.wrap
def fused_mul_sum_dim1_unsqueeze_sigmoid(x, y):
    n = x.shape[0]
    h = x.shape[2]
    w = x.shape[3]

    out = torch.empty((n, 1, h, w), device=x.device, dtype=x.dtype)

    if x.dtype == torch.float32:
        if n == 1:
            BLOCK_HW = 1024
            BLOCK_C = 8
            num_warps = 8
        else:
            BLOCK_HW = 512
            BLOCK_C = 8
            num_warps = 8
        grid = (4096 // BLOCK_HW, n)
        fused_mul_sum_sigmoid_kernel_f32[grid](
            x,
            y,
            out,
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        if n == 1:
            BLOCK_HW = 1024
            BLOCK_C = 8
            num_warps = 8
        elif n == 8:
            BLOCK_HW = 512
            BLOCK_C = 8
            num_warps = 8
        else:
            BLOCK_HW = 256
            BLOCK_C = 8
            num_warps = 4
        grid = (4096 // BLOCK_HW, n)
        fused_mul_sum_sigmoid_kernel_lowp[grid](
            x,
            y,
            out,
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=2,
        )
    return out


def replacement_func():
    return fused_mul_sum_dim1_unsqueeze_sigmoid