import torch
import triton
import triton.language as tl


@triton.jit
def _roll_to_seq_kernel(
    in3_ptr,
    out_ptr,
    H,
    W,
    C,
    INNER_H,
    INNER_W,
    SHIFT_H,
    SHIFT_W,
    stride_in3_d1,
    stride_in3_d2,
    stride_in3_d3,
    stride_in3_d4,
    stride_in3_d5,
    stride_out_d1,
    stride_out_d2,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    h = pid_hw // W
    w = pid_hw % W
    src_h = (h - SHIFT_H + H) % H
    src_w = (w - SHIFT_W + W) % W

    d1 = src_h // INNER_H
    d2 = src_h % INNER_H
    d3 = src_w // INNER_W
    d4 = src_w % INNER_W

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < C

    in3_base = d1 * stride_in3_d1 + d2 * stride_in3_d2 + d3 * stride_in3_d3 + d4 * stride_in3_d4
    x = tl.load(in3_ptr + in3_base + offs_c * stride_in3_d5, mask=mask, other=0.0)
    tl.store(out_ptr + pid_hw * stride_out_d1 + offs_c * stride_out_d2, x, mask=mask)


@torch.fx.wrap
def roll_to_seq_dispatch(in_3, route):
    if route == 's32_c768':
        H = 32
        W = 32
        C = 768
        INNER_H = 8
        INNER_W = 8
    elif route == 's64_c384':
        H = 64
        W = 64
        C = 384
        INNER_H = 8
        INNER_W = 8
    else:
        raise RuntimeError(f'Unknown route: {route}')

    out = torch.empty((1, H * W, C), device=in_3.device, dtype=in_3.dtype)
    s_in3 = in_3.stride()
    s_out = out.stride()
    grid = (H * W, triton.cdiv(C, 256))

    _roll_to_seq_kernel[grid](
        in_3,
        out,
        H,
        W,
        C,
        INNER_H,
        INNER_W,
        4,
        4,
        s_in3[1],
        s_in3[2],
        s_in3[3],
        s_in3[4],
        s_in3[5],
        s_out[1],
        s_out[2],
        BLOCK_C=256,
    )
    return out