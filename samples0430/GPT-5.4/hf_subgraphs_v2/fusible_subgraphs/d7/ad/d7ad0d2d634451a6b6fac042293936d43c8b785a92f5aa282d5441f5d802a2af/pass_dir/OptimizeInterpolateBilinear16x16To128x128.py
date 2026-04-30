import torch
import triton
import triton.language as tl


def pattern(x):
    return torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4, num_stages=2),
    ],
    key=['C'],
)
@triton.jit

def _upsample_bilinear_16_to_128_kernel(
    x_ptr,
    out_ptr,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    o_s0,
    o_s1,
    o_s2,
    o_s3,
    C,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    c = pid_bc % C
    b = pid_bc // C

    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    fy = (oh.to(tl.float32) + 0.5) * 0.125 - 0.5
    fx = (ow.to(tl.float32) + 0.5) * 0.125 - 0.5

    fy = tl.minimum(tl.maximum(fy, 0.0), 15.0)
    fx = tl.minimum(tl.maximum(fx, 0.0), 15.0)

    y0 = tl.floor(fy).to(tl.int32)
    x0 = tl.floor(fx).to(tl.int32)
    y1 = tl.minimum(y0 + 1, 15)
    x1 = tl.minimum(x0 + 1, 15)

    wy = fy - y0.to(tl.float32)
    wx = fx - x0.to(tl.float32)
    hy = 1.0 - wy
    hx = 1.0 - wx

    base_in = x_ptr + b * x_s0 + c * x_s1

    off00 = y0[:, None] * x_s2 + x0[None, :] * x_s3
    off01 = y0[:, None] * x_s2 + x1[None, :] * x_s3
    off10 = y1[:, None] * x_s2 + x0[None, :] * x_s3
    off11 = y1[:, None] * x_s2 + x1[None, :] * x_s3

    v00 = tl.load(base_in + off00).to(tl.float32)
    v01 = tl.load(base_in + off01).to(tl.float32)
    v10 = tl.load(base_in + off10).to(tl.float32)
    v11 = tl.load(base_in + off11).to(tl.float32)

    w00 = hy[:, None] * hx[None, :]
    w01 = hy[:, None] * wx[None, :]
    w10 = wy[:, None] * hx[None, :]
    w11 = wy[:, None] * wx[None, :]

    out = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11

    base_out = out_ptr + b * o_s0 + c * o_s1
    tl.store(base_out + oh[:, None] * o_s2 + ow[None, :] * o_s3, out)


@torch.fx.wrap
def triton_upsample_bilinear_16x16_to_128x128(x):
    b = x.shape[0]
    c = x.shape[1]
    out = torch.empty((b, c, 128, 128), device=x.device, dtype=x.dtype)

    x_s0, x_s1, x_s2, x_s3 = x.stride()
    o_s0, o_s1, o_s2, o_s3 = out.stride()

    grid = lambda meta: (b * c, 128 // meta['BLOCK_H'], 128 // meta['BLOCK_W'])

    _upsample_bilinear_16_to_128_kernel[grid](
        x,
        out,
        x_s0,
        x_s1,
        x_s2,
        x_s3,
        o_s0,
        o_s1,
        o_s2,
        o_s3,
        c,
    )
    return out


def replacement_func():
    return triton_upsample_bilinear_16x16_to_128x128