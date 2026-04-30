import torch
import triton
import triton.language as tl


def pattern(in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(in_4, in_5):
    return (in_4, in_5)


@triton.jit
def fused_add_mean_small_spatial_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    sxn,
    sxc,
    sxh,
    sxw,
    syn,
    syc,
    syh,
    syw,
    so0,
    so1,
    inv_hw,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    hw = H * W
    offs = tl.arange(0, BLOCK_HW)
    h_idx = offs // W
    w_idx = offs % W
    mask = offs < hw

    x_base = n * sxn + c * sxc
    y_base = n * syn + c * syc

    x_vals = tl.load(x_ptr + x_base + h_idx * sxh + w_idx * sxw, mask=mask, other=0).to(tl.float32)
    y_vals = tl.load(y_ptr + y_base + h_idx * syh + w_idx * syw, mask=mask, other=0).to(tl.float32)
    mean_val = tl.sum(x_vals + y_vals, axis=0) * inv_hw

    tl.store(out_ptr + n * so0 + c * so1, mean_val)


@torch.fx.wrap
def fused_add_mean_small_spatial(x, y):
    N, C, H, W = x.shape
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)

    hw = H * W
    if hw <= 64:
        block_hw = 64
        num_warps = 2
    elif hw <= 128:
        block_hw = 128
        num_warps = 4
    else:
        block_hw = 256
        num_warps = 8

    grid = (N * C,)
    fused_add_mean_small_spatial_kernel[grid](
        x,
        y,
        out,
        N,
        C,
        H,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        out.stride(0),
        out.stride(1),
        1.0 / float(hw),
        BLOCK_HW=block_hw,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_add_mean_small_spatial