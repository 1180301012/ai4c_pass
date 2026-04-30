import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    ],
    key=["OUT_C", "NPOS", "IN_C"],
)
@triton.jit
def pointwise_conv_bn_kernel(
    x_ptr,
    w_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    OUT_C,
    NPOS,
    IN_C,
    HW,
    WDIM,
    sxb,
    sxc,
    sxh,
    sxw,
    sw0,
    sw1,
    sob,
    soc,
    soh,
    sow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_b = offs_n // HW
    offs_hw = offs_n % HW
    offs_h = offs_hw // WDIM
    offs_w = offs_hw % WDIM

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_C, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w_ptrs = w_ptr + offs_m[:, None] * sw0 + offs_k[None, :] * sw1
        x_ptrs = (
            x_ptr
            + offs_b[None, :] * sxb
            + offs_k[:, None] * sxc
            + offs_h[None, :] * sxh
            + offs_w[None, :] * sxw
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_m[:, None] < OUT_C) & (offs_k[None, :] < IN_C),
            other=0.0,
        )
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < IN_C) & (offs_n[None, :] < NPOS),
            other=0.0,
        )
        acc += tl.dot(w, x)

    mean = tl.load(mean_ptr + offs_m, mask=offs_m < OUT_C, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + offs_m, mask=offs_m < OUT_C, other=1.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_m, mask=offs_m < OUT_C, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_m, mask=offs_m < OUT_C, other=0.0).to(tl.float32)

    scale = gamma * tl.rsqrt(var + 1e-5)
    bias = beta - mean * scale
    out = acc * scale[:, None] + bias[:, None]

    out_ptrs = (
        out_ptr
        + offs_b[None, :] * sob
        + offs_m[:, None] * soc
        + offs_h[None, :] * soh
        + offs_w[None, :] * sow
    )
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < OUT_C) & (offs_n[None, :] < NPOS))


@torch.fx.wrap
def fused_pointwise_conv_bn(x, weight, running_mean, running_var, gamma, beta):
    bsz, _, hdim, wdim = x.shape
    out_c = weight.shape[0]
    hw = hdim * wdim
    npos = bsz * hw
    out = torch.empty((bsz, out_c, hdim, wdim), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(out_c, META["BLOCK_M"]),
        triton.cdiv(npos, META["BLOCK_N"]),
    )

    pointwise_conv_bn_kernel[grid](
        x,
        weight,
        running_mean,
        running_var,
        gamma,
        beta,
        out,
        out_c,
        npos,
        x.shape[1],
        hw,
        wdim,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return fused_pointwise_conv_bn