import torch
import triton
import triton.language as tl


@triton.jit
def _mean_seq49_kernel(
    x_ptr,
    out_ptr,
    x_s0,
    x_s1,
    x_s2,
    out_s0,
    out_s1,
    D,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    base_ptr = x_ptr + pid_b * x_s0 + offs_d * x_s2
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in range(49):
        v = tl.load(base_ptr + s * x_s1, mask=mask_d, other=0.0)
        acc += v.to(tl.float32)
    tl.store(out_ptr + pid_b * out_s0 + offs_d * out_s1, acc * (1.0 / 49.0), mask=mask_d)


@triton.jit
def _linear_2x448_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    x_s0,
    x_s1,
    w_s0,
    w_s1,
    out_s0,
    out_s1,
    K,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)

    acc0 = 0.0
    acc1 = 0.0
    for k_start in range(0, 448, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(x_ptr + pid_b * x_s0 + offs_k * x_s1, mask=mask_k, other=0.0).to(tl.float32)
        w0 = tl.load(weight_ptr + offs_k * w_s1, mask=mask_k, other=0.0).to(tl.float32)
        w1 = tl.load(weight_ptr + w_s0 + offs_k * w_s1, mask=mask_k, other=0.0).to(tl.float32)
        acc0 += tl.sum(x * w0, axis=0)
        acc1 += tl.sum(x * w1, axis=0)

    b0 = tl.load(bias_ptr + 0).to(tl.float32)
    b1 = tl.load(bias_ptr + 1).to(tl.float32)
    tl.store(out_ptr + pid_b * out_s0, acc0 + b0)
    tl.store(out_ptr + pid_b * out_s0 + out_s1, acc1 + b1)


@torch.fx.wrap
def classifier_head_dispatch(*args):
    route = args[-1]
    if route == "mean":
        x = args[0]
        B = x.shape[0]
        D = x.shape[2]
        out = torch.empty((B, D), device=x.device, dtype=x.dtype)
        grid = (B, triton.cdiv(D, 128))
        _mean_seq49_kernel[grid](
            x,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out.stride(0),
            out.stride(1),
            D,
            BLOCK_D=128,
            num_warps=4,
            num_stages=2,
        )
        return out

    if route == "linear":
        bias, weight, x = args[0], args[1], args[2]
        B = x.shape[0]
        out = torch.empty((B, 2), device=x.device, dtype=x.dtype)
        _linear_2x448_kernel[(B,)](
            bias,
            weight,
            x,
            out,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            x.shape[1],
            BLOCK_K=128,
            num_warps=2,
            num_stages=2,
        )
        return out

    raise RuntimeError(f"Unknown route: {route}")