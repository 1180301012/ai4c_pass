import torch
import triton
import triton.language as tl


@triton.jit
def unpack_bhsd_to_bsd_kernel(
    x_ptr,
    out_ptr,
    B,
    H,
    S,
    D,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_x3,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_row // S
    s = pid_row % S

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (b < B) & (d_offsets < D)

    h = d_offsets // 64
    dh = d_offsets % 64

    x_ptrs = x_ptr + b * stride_x0 + h * stride_x1 + s * stride_x2 + dh * stride_x3
    vals = tl.load(x_ptrs, mask=mask, other=0.0)

    out_ptrs = out_ptr + b * stride_o0 + s * stride_o1 + d_offsets * stride_o2
    tl.store(out_ptrs, vals, mask=mask)


@torch.fx.wrap
def sdpa_unpack_last(x, route):
    # route is present to keep a stable shared replacement_func() across passes.
    B = x.shape[0]
    H = x.shape[1]
    S = x.shape[2]
    DH = x.shape[3]
    D = H * DH

    out = torch.empty((B, S, D), device=x.device, dtype=x.dtype)

    if D <= 128:
        BLOCK_D = 128
        num_warps = 2
    elif D <= 256:
        BLOCK_D = 256
        num_warps = 4
    else:
        BLOCK_D = 256
        num_warps = 8

    grid = (B * S, triton.cdiv(D, BLOCK_D))
    unpack_bhsd_to_bsd_kernel[grid](
        x,
        out,
        B,
        H,
        S,
        D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out


def replacement_func():
    return sdpa_unpack_last