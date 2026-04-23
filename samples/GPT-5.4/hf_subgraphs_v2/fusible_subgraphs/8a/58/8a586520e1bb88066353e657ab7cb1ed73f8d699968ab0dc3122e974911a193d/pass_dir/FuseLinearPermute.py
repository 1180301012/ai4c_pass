import torch
import triton
import triton.language as tl

from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


# Pattern matching function
# Mirrors the compute branch exactly: linear -> permute
def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3


# Argument extraction function
def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_POS": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_POS": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_POS": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_POS": 512}, num_warps=8, num_stages=2),
    ],
    key=["P"],
)
@triton.jit
def _linear3_perm_to_nchw_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    w_s0,
    w_s1,
    b_s0,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    P,
    BLOCK_POS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_p = pid * BLOCK_POS + tl.arange(0, BLOCK_POS)
    mask_p = offs_p < P

    n = offs_p % N
    tmp = offs_p // N
    m = tmp % M
    b = tmp // M

    base_x = b * x_s0 + m * x_s1 + n * x_s2

    x0 = tl.load(x_ptr + base_x + 0 * x_s3, mask=mask_p, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base_x + 1 * x_s3, mask=mask_p, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + base_x + 2 * x_s3, mask=mask_p, other=0.0).to(tl.float32)

    offs_o = tl.arange(0, 16)
    w0 = tl.load(w_ptr + offs_o * w_s0 + 0 * w_s1).to(tl.float32)
    w1 = tl.load(w_ptr + offs_o * w_s0 + 1 * w_s1).to(tl.float32)
    w2 = tl.load(w_ptr + offs_o * w_s0 + 2 * w_s1).to(tl.float32)
    bias = tl.load(b_ptr + offs_o * b_s0).to(tl.float32)

    acc = bias[None, :]
    acc += x0[:, None] * w0[None, :]
    acc += x1[:, None] * w1[None, :]
    acc += x2[:, None] * w2[None, :]

    out_ptrs = (
        out_ptr
        + b[:, None] * out_s0
        + offs_o[None, :] * out_s1
        + m[:, None] * out_s2
        + n[:, None] * out_s3
    )
    tl.store(out_ptrs, acc, mask=mask_p[:, None])


# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_linear_permute(in_0, in_1, in_3):
    b = unwrap_tensor(in_0)
    w = unwrap_tensor(in_1)
    x = unwrap_tensor(in_3)

    B = x.shape[0]
    M = x.shape[1]
    N = x.shape[2]

    out = torch.empty((B, 16, M, N), device=x.device, dtype=x.dtype)

    P = B * M * N
    grid = lambda META: (triton.cdiv(P, META["BLOCK_POS"]),)

    _linear3_perm_to_nchw_kernel[grid](
        x,
        w,
        b,
        out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        w.stride(0),
        w.stride(1),
        b.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        P,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_linear_permute