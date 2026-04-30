import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _masked_mean_pool_broadcast_lastdim0_kernel(
    mask_ptr,
    x_ptr,
    out_ptr,
    mask_s0,
    mask_s1,
    x_s0,
    x_s1,
    x_s2,
    H,
    S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    denom = 0.0

    mask_base = mask_ptr + pid_b * mask_s0
    x_base = x_ptr + pid_b * x_s0

    for s in tl.static_range(0, S):
        mask_val = tl.load(mask_base + s * mask_s1).to(tl.float32)
        x_val = tl.load(x_base + s * x_s1 + offs_h * x_s2, mask=h_mask, other=0.0).to(tl.float32)
        acc += x_val * mask_val
        denom += mask_val

    denom = tl.maximum(denom, 1e-09)
    out_val = acc / denom
    tl.store(out_ptr + pid_b * H + offs_h, out_val, mask=h_mask)


@triton.jit
def _masked_mean_pool_generic_kernel(
    mask_ptr,
    x_ptr,
    out_ptr,
    mask_s0,
    mask_s1,
    mask_s2,
    x_s0,
    x_s1,
    x_s2,
    H,
    S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    denom = tl.zeros((BLOCK_H,), dtype=tl.float32)

    mask_base = mask_ptr + pid_b * mask_s0
    x_base = x_ptr + pid_b * x_s0

    for s in tl.static_range(0, S):
        mask_val = tl.load(mask_base + s * mask_s1 + offs_h * mask_s2, mask=h_mask, other=0).to(tl.float32)
        x_val = tl.load(x_base + s * x_s1 + offs_h * x_s2, mask=h_mask, other=0.0).to(tl.float32)
        acc += x_val * mask_val
        denom += mask_val

    denom = tl.maximum(denom, 1e-09)
    out_val = acc / denom
    tl.store(out_ptr + pid_b * H + offs_h, out_val, mask=h_mask)


@torch.fx.wrap
def fused_masked_mean_pool_sum_div_cat_dim1(in_0, in_1):
    B = in_1.shape[0]
    S = in_1.shape[1]
    H = in_1.shape[2]

    out = torch.empty((B, H), device=in_1.device, dtype=torch.float32)

    if H <= 128:
        BLOCK_H = 128
        num_warps = 4
    elif H <= 256:
        BLOCK_H = 256
        num_warps = 4
    elif H <= 512:
        BLOCK_H = 512
        num_warps = 8
    else:
        BLOCK_H = 256
        num_warps = 4

    grid = (B, triton.cdiv(H, BLOCK_H))

    if in_0.stride(2) == 0:
        _masked_mean_pool_broadcast_lastdim0_kernel[grid](
            in_0,
            in_1,
            out,
            in_0.stride(0),
            in_0.stride(1),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
            H,
            S=S,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _masked_mean_pool_generic_kernel[grid](
            in_0,
            in_1,
            out,
            in_0.stride(0),
            in_0.stride(1),
            in_0.stride(2),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
            H,
            S=S,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=1,
        )

    return out


def replacement_func():
    return fused_masked_mean_pool_sum_div_cat_dim1