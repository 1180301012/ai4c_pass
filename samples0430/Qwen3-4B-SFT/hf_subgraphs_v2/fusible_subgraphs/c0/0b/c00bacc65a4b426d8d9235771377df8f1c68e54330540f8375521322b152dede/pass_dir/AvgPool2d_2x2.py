import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['HW'],
)
@triton.jit
def avg_pool2d_2x2_kernel(
    in_ptr,
    out_ptr,
    HW,
    H,
    W,
    H_out,
    W_out,
    W_out_int,     # = W_out as int (for address computation; avoids mixed-type arithmetic issues)
    BLOCK_SIZE: tl.constexpr,
):
    """
    2x2 average pooling with stride=2, count_include_pad=True.
    Grid: dim0 = N*C, dim1 = ceil(H_out * W_out / BLOCK_SIZE).
    """
    nc_flat = tl.program_id(0)
    tile_id  = tl.program_id(1)

    h_out = tl.arange(0, BLOCK_SIZE) // W_out_int
    w_out = tl.arange(0, BLOCK_SIZE) % W_out_int
    mask2d = (h_out < H_out) & (w_out < W_out_int)

    acc  = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    base = nc_flat * HW
    h0 = h_out * 2
    w0 = w_out * 2

    x00 = tl.load(in_ptr + base + h0  * W + w0,
                  mask=mask2d & (h0  < H) & (w0  < W), other=0.0).to(tl.float32)
    x01 = tl.load(in_ptr + base + h0  * W + (w0 + 1),
                  mask=mask2d & (h0  < H) & ((w0 + 1) < W), other=0.0).to(tl.float32)
    x10 = tl.load(in_ptr + base + (h0 + 1) * W + w0,
                  mask=mask2d & ((h0 + 1) < H) & (w0  < W), other=0.0).to(tl.float32)
    x11 = tl.load(in_ptr + base + (h0 + 1) * W + (w0 + 1),
                  mask=mask2d & ((h0 + 1) < H) & ((w0 + 1) < W), other=0.0).to(tl.float32)

    acc = x00 + x01 + x10 + x11
    tl.store(
        out_ptr + base + h_out * W_out_int + w_out,
        acc * 0.25,
        mask=mask2d,
    )


@torch.fx.wrap
def triton_avg_pool2d_2x2(x):
    """
    Replace torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)
    with an optimized Triton kernel for 2x2 stride-2 average pooling.
    """
    N, C, H, W = x.shape
    NC   = N * C
    H_out = H // 2
    W_out = W // 2
    HW   = H * W

    out = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)

    def grid(meta):
        return (NC, (H_out * W_out + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    avg_pool2d_2x2_kernel[grid](
        x,
        out,
        HW,
        H,
        W,
        H_out,
        W_out,
        W_out,       # W_out_int copy – needed as runtime int in kernel
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(x):
    result = torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)
    return result


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_avg_pool2d_2x2