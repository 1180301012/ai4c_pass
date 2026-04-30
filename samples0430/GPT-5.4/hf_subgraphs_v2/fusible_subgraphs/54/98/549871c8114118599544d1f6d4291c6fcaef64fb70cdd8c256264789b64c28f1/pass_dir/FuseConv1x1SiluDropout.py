import torch
import triton
import triton.language as tl
from pass_dir.shared_zero_dispatch import shared_zero_dispatch


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.silu(conv2d, inplace=False)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0, "conv_like")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_OC": 64, "BLOCK_P": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_OC": 64, "BLOCK_P": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_OC": 128, "BLOCK_P": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_OC": 64, "BLOCK_P": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_OC": 128, "BLOCK_P": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    ],
    key=["BATCH", "OUT_C", "P", "K"],
)
@triton.jit
def fused_conv1x1_silu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    BATCH,
    OUT_C,
    P,
    K,
    x_stride_n,
    x_stride_c,
    w_stride_oc,
    w_stride_ic,
    out_stride_n,
    out_stride_c,
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_p = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        w_ptrs = w_ptr + offs_oc[:, None] * w_stride_oc + k_idx[None, :] * w_stride_ic
        x_ptrs = x_ptr + pid_n * x_stride_n + k_idx[:, None] * x_stride_c + offs_p[None, :]

        w = tl.load(
            w_ptrs,
            mask=(offs_oc[:, None] < OUT_C) & (k_idx[None, :] < K),
            other=0.0,
        )
        x = tl.load(
            x_ptrs,
            mask=(k_idx[:, None] < K) & (offs_p[None, :] < P),
            other=0.0,
        )
        acc += tl.dot(w, x)

    bias = tl.load(b_ptr + offs_oc, mask=offs_oc < OUT_C, other=0.0).to(tl.float32)
    acc += bias[:, None]
    acc = acc * tl.sigmoid(acc)

    out_ptrs = out_ptr + pid_n * out_stride_n + offs_oc[:, None] * out_stride_c + offs_p[None, :]
    tl.store(out_ptrs, acc, mask=(offs_oc[:, None] < OUT_C) & (offs_p[None, :] < P))


@torch.fx.wrap
def fused_conv1x1_silu_dropout(x, weight, bias):
    batch = x.shape[0]
    h = x.shape[2]
    w = x.shape[3]
    cin = x.shape[1]
    cout = weight.shape[0]
    p = h * w

    out = torch.empty((batch, cout, h, w), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(p, META["BLOCK_P"]),
        triton.cdiv(cout, META["BLOCK_OC"]),
        batch,
    )

    fused_conv1x1_silu_kernel[grid](
        x_ptr=x,
        w_ptr=weight,
        b_ptr=bias,
        out_ptr=out,
        BATCH=batch,
        OUT_C=cout,
        P=p,
        K=cin,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        w_stride_oc=weight.stride(0),
        w_stride_ic=weight.stride(1),
        out_stride_n=out.stride(0),
        out_stride_c=out.stride(1),
    )
    return out


def replacement_func():
    return shared_zero_dispatch