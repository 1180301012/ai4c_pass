import torch
import triton
import triton.language as tl

CIN = 160
COUT = 17
HW = 64 * 48


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["BATCH"],
)
@triton.jit
def conv1x1_flatten_17x160_hw3072_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    BATCH,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < 3072
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, 16)

    x_batch_ptr = x_ptr + pid_b * 160 * 3072
    out_batch_ptr = out_ptr + pid_b * 17 * 3072

    acc0 = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    bias0 = tl.load(b_ptr + offs_n).to(tl.float32)
    bias1 = tl.load(b_ptr + 16).to(tl.float32)

    for k0 in tl.static_range(0, 160, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            x_batch_ptr + k[:, None] * 3072 + offs_m[None, :],
            mask=mask_m[None, :],
            other=0.0,
        )
        w0 = tl.load(w_ptr + k[:, None] + offs_n[None, :] * 160)
        x_t = tl.trans(x)
        acc0 += tl.dot(x_t, w0)

        w1 = tl.load(w_ptr + 16 * 160 + k).to(tl.float32)
        acc1 += tl.sum(x_t.to(tl.float32) * w1[None, :], axis=1)

    acc0 += bias0[None, :]
    acc1 += bias1

    out0_ptrs = out_batch_ptr + offs_m[:, None] + offs_n[None, :] * 3072
    tl.store(out0_ptrs, acc0, mask=mask_m[:, None])

    out1_ptrs = out_batch_ptr + 16 * 3072 + offs_m
    tl.store(out1_ptrs, acc1, mask=mask_m)


@torch.fx.wrap
def triton_conv1x1_flatten_17x160_hw3072(bias, weight, x):
    batch = x.shape[0]
    out = torch.empty((batch, COUT, HW), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(HW, meta["BLOCK_M"]), batch)
    conv1x1_flatten_17x160_hw3072_kernel[grid](
        x,
        weight,
        bias,
        out,
        batch,
    )
    return out


def replacement_func():
    return triton_conv1x1_flatten_17x160_hw3072