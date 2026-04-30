import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_1_2_kernel(
    x_ptr,
    out_ptr,
    batch,
    m_size,
    n_size,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)

    x_ptrs = x_ptr + pid_b * stride_xb + offs_n[None, :] * stride_xm + offs_m[:, None] * stride_xn
    vals = tl.load(x_ptrs, mask=mask, other=0)

    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def _pack_heads9_kernel(
    x_ptr,
    out_ptr,
    l_size,
    groups,
    heads,
    stride_xn,
    stride_xl,
    packed_size,
    BLOCK_PACKED: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_p = tl.program_id(1)

    offs_p = pid_p * BLOCK_PACKED + tl.arange(0, BLOCK_PACKED)
    mask_p = offs_p < packed_size
    safe_p = tl.where(mask_p, offs_p, 0)

    g = safe_p // 9
    k = safe_p - g * 9

    l = pid_d // heads
    h = pid_d - l * heads
    c = h * groups + g
    n = c * 9 + k

    vals = tl.load(x_ptr + n * stride_xn + l * stride_xl, mask=mask_p, other=0)
    tl.store(out_ptr + pid_d * packed_size + offs_p, vals, mask=mask_p)


@torch.fx.wrap
def dispatch_convbert_replacement(x, route):
    if route == "transpose":
        batch = x.shape[0]
        n_size = x.shape[1]
        m_size = x.shape[2]
        out = torch.empty((batch, m_size, n_size), device=x.device, dtype=x.dtype)

        if n_size >= 1024:
            block_m = 32
            block_n = 32
            num_warps = 4
        elif n_size >= 256:
            block_m = 32
            block_n = 16
            num_warps = 2
        else:
            block_m = 16
            block_n = 16
            num_warps = 1

        grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n), batch)
        _transpose_1_2_kernel[grid](
            x,
            out,
            batch,
            m_size,
            n_size,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
        )
        return out

    if route == "pack_16_8":
        groups = 8
        heads = 2
    elif route == "pack_384_64":
        groups = 64
        heads = 6
    else:
        return x

    l_size = x.shape[2]
    d_size = l_size * heads
    packed_size = groups * 9
    out = torch.empty((d_size, groups, 9), device=x.device, dtype=x.dtype)

    if packed_size <= 128:
        block_packed = 128
        num_warps = 1
    elif packed_size <= 256:
        block_packed = 256
        num_warps = 2
    else:
        block_packed = 256
        num_warps = 4

    grid = (d_size, triton.cdiv(packed_size, block_packed))
    _pack_heads9_kernel[grid](
        x,
        out,
        l_size,
        groups,
        heads,
        x.stride(1),
        x.stride(2),
        packed_size,
        BLOCK_PACKED=block_packed,
        num_warps=num_warps,
        num_stages=1,
    )
    return out