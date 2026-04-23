import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_CO": 32, "BLOCK_HW": 64, "BLOCK_CI": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_CO": 32, "BLOCK_HW": 128, "BLOCK_CI": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_CO": 64, "BLOCK_HW": 64, "BLOCK_CI": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_CO": 64, "BLOCK_HW": 128, "BLOCK_CI": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_CO": 64, "BLOCK_HW": 64, "BLOCK_CI": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_CO": 64, "BLOCK_HW": 128, "BLOCK_CI": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_CO": 128, "BLOCK_HW": 64, "BLOCK_CI": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_CO": 128, "BLOCK_HW": 128, "BLOCK_CI": 32}, num_warps=8, num_stages=2),
    ],
    key=["COUT", "HW", "CIN", "DTYPE_ID"],
)
@triton.jit
def fused_conv1x1_bn_add_kernel(
    x_ptr,
    w_ptr,
    rm_ptr,
    rv_ptr,
    beta_ptr,
    gamma_ptr,
    residual_ptr,
    out_ptr,
    B,
    CIN,
    COUT,
    H,
    W,
    x_sN,
    x_sC,
    x_sH,
    x_sW,
    w_sO,
    w_sI,
    rm_s0,
    rv_s0,
    beta_s0,
    gamma_s0,
    residual_sN,
    residual_sC,
    residual_sH,
    residual_sW,
    out_sN,
    out_sC,
    out_sH,
    out_sW,
    HW,
    DTYPE_ID,
    BLOCK_CO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid_co = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    hw_mask = offs_hw < HW
    co_mask = offs_co < COUT

    offs_h = offs_hw // W
    offs_w = offs_hw - offs_h * W

    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    offs_ci_base = tl.arange(0, BLOCK_CI)
    for ci_start in range(0, CIN, BLOCK_CI):
        offs_ci = ci_start + offs_ci_base
        ci_mask = offs_ci < CIN

        x_ptrs = (
            x_ptr
            + pid_b * x_sN
            + offs_ci[:, None] * x_sC
            + offs_h[None, :] * x_sH
            + offs_w[None, :] * x_sW
        )
        w_ptrs = w_ptr + offs_co[:, None] * w_sO + offs_ci[None, :] * w_sI

        x = tl.load(x_ptrs, mask=ci_mask[:, None] & hw_mask[None, :], other=0.0)
        wv = tl.load(w_ptrs, mask=co_mask[:, None] & ci_mask[None, :], other=0.0)

        acc += tl.dot(wv.to(tl.float32), x.to(tl.float32))

    rm = tl.load(rm_ptr + offs_co * rm_s0, mask=co_mask, other=0.0).to(tl.float32)
    rv = tl.load(rv_ptr + offs_co * rv_s0, mask=co_mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_co * beta_s0, mask=co_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_co * gamma_s0, mask=co_mask, other=1.0).to(tl.float32)

    scale = gamma * tl.rsqrt(rv + 1.0e-5)
    bias = beta - rm * scale

    residual_ptrs = (
        residual_ptr
        + pid_b * residual_sN
        + offs_co[:, None] * residual_sC
        + offs_h[None, :] * residual_sH
        + offs_w[None, :] * residual_sW
    )
    residual = tl.load(residual_ptrs, mask=co_mask[:, None] & hw_mask[None, :], other=0.0).to(tl.float32)

    out = acc * scale[:, None] + bias[:, None] + residual

    out_ptrs = (
        out_ptr
        + pid_b * out_sN
        + offs_co[:, None] * out_sC
        + offs_h[None, :] * out_sH
        + offs_w[None, :] * out_sW
    )
    tl.store(out_ptrs, out, mask=co_mask[:, None] & hw_mask[None, :])


@torch.fx.wrap
def dispatch_fused_conv_bn_add(rm, rv, beta, gamma, w, x, residual):
    B = x.shape[0]
    CIN = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    COUT = w.shape[0]
    HW = H * W

    out = torch.empty((B, COUT, H, W), device=x.device, dtype=x.dtype)

    if x.dtype == torch.float16:
        dtype_id = 0
    elif x.dtype == torch.bfloat16:
        dtype_id = 1
    else:
        dtype_id = 2

    grid = lambda meta: (
        triton.cdiv(COUT, meta["BLOCK_CO"]),
        triton.cdiv(HW, meta["BLOCK_HW"]),
        B,
    )

    fused_conv1x1_bn_add_kernel[grid](
        x,
        w,
        rm,
        rv,
        beta,
        gamma,
        residual,
        out,
        B,
        CIN,
        COUT,
        H,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        w.stride(0),
        w.stride(1),
        rm.stride(0),
        rv.stride(0),
        beta.stride(0),
        gamma.stride(0),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        residual.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        HW,
        dtype_id,
    )
    return (out,)


def shared_replacement_func():
    return dispatch_fused_conv_bn_add