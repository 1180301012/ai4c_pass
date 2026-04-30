import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 / 8.0
    tmp_0 += in_2
    tmp_1 = tmp_0
    tmp_2 = tmp_1 + in_1
    return (tmp_2,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_div_add_add_broadcast_kernel(
    x_ptr,
    mask_ptr,
    v_ptr,
    out_ptr,
    sx0,
    sx1,
    sx2,
    sx3,
    sm0,
    sm1,
    sm2,
    sm3,
    sv0,
    sv1,
    sv2,
    sv3,
    B,
    H,
    M,
    N,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < numel

    n = offs % N
    t = offs // N
    m = t % M
    t = t // M
    h = t % H
    b = t // H

    x_idx = b * sx0 + h * sx1 + m * sx2 + n * sx3
    v_idx = b * sv0 + h * sv1 + m * sv2 + n * sv3
    mask_idx = b * sm0 + n * sm3

    x = tl.load(x_ptr + x_idx, mask=valid, other=0)
    v = tl.load(v_ptr + v_idx, mask=valid, other=0)
    mask_v = tl.load(mask_ptr + mask_idx, mask=valid, other=0)

    tmp = tl.cast(x.to(tl.float32) * 0.125, x.dtype)
    tmp = tl.cast(tmp.to(tl.float32) + v.to(tl.float32), x.dtype)
    out = tl.cast(tmp.to(tl.float32) + mask_v.to(tl.float32), x.dtype)

    tl.store(out_ptr + offs, out, mask=valid)


@torch.fx.wrap
def fused_div_add_add_broadcast(in_0, in_1, in_2):
    B = in_0.shape[0]
    H = in_0.shape[1]
    M = in_0.shape[2]
    N = in_0.shape[3]
    numel = in_0.numel()

    out = torch.empty(in_0.shape, device=in_0.device, dtype=in_0.dtype)

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_div_add_add_broadcast_kernel[grid](
        in_0,
        in_1,
        in_2,
        out,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        B,
        H,
        M,
        N,
        numel,
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=1,
    )
    return (out,)


def replacement_func():
    return fused_div_add_add_broadcast