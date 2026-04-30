import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8),
    ],
    key=['tokens', 'dmodel'],
)
@triton.jit
def _transpose_mul_kernel(
    tmp4_ptr,
    in6_ptr,
    out_ptr,
    batch,
    heads,
    channels,
    tokens,
    dmodel,
    stride_tmp4_b,
    stride_tmp4_h,
    stride_tmp4_c,
    stride_tmp4_t,
    stride_in6_b,
    stride_in6_h,
    stride_in6_t,
    stride_in6_c,
    stride_out_b,
    stride_out_h,
    stride_out_t,
    stride_out_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    t = offs_m[:, None]
    d = offs_n[None, :]

    h = pid_bh % heads
    b = pid_bh // heads
    c = d

    mask = (b < batch) & (t < tokens) & (d < dmodel)

    tmp4_ptrs = (
        tmp4_ptr
        + b * stride_tmp4_b
        + h * stride_tmp4_h
        + c * stride_tmp4_c
        + t * stride_tmp4_t
    )
    in6_ptrs = (
        in6_ptr
        + b * stride_in6_b
        + h * stride_in6_h
        + t * stride_in6_t
        + c * stride_in6_c
    )

    x = tl.load(tmp4_ptrs, mask=mask, other=0).to(tl.float32)
    y = tl.load(in6_ptrs, mask=mask, other=0).to(tl.float32)
    z = x * y

    out_ptrs = (
        out_ptr
        + b * stride_out_b
        + h * stride_out_h
        + t * stride_out_t
        + c * stride_out_c
    )
    tl.store(out_ptrs, z, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 32, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_D': 64}, num_warps=8),
        triton.Config({'BLOCK_T': 128, 'BLOCK_D': 32}, num_warps=8),
    ],
    key=['dim1', 'dim2'],
)
@triton.jit
def _scale_add_transpose_reshape_kernel(
    in4_ptr,
    padded_ptr,
    out_ptr,
    scale,
    batch,
    heads,
    channels,
    dim1,
    dim2,
    stride_in4_b,
    stride_in4_h,
    stride_in4_t,
    stride_in4_c,
    stride_pad_b,
    stride_pad_h,
    stride_pad_t,
    stride_pad_c,
    stride_out_b,
    stride_out_t,
    stride_out_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    t = offs_t[:, None]
    d = offs_d[None, :]
    h = d // channels
    c = d - h * channels

    mask = (pid_b < batch) & (t < dim1) & (d < dim2)

    in4_ptrs = (
        in4_ptr
        + pid_b * stride_in4_b
        + h * stride_in4_h
        + t * stride_in4_t
        + c * stride_in4_c
    )
    pad_ptrs = (
        padded_ptr
        + pid_b * stride_pad_b
        + h * stride_pad_h
        + t * stride_pad_t
        + c * stride_pad_c
    )
    out_ptrs = (
        out_ptr
        + pid_b * stride_out_b
        + t * stride_out_t
        + d * stride_out_d
    )

    a = tl.load(in4_ptrs, mask=mask, other=0).to(tl.float32)
    b = tl.load(pad_ptrs, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptrs, a * scale + b, mask=mask)



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 32, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_T': 64, 'BLOCK_D': 64}, num_warps=8),
        triton.Config({'BLOCK_T': 128, 'BLOCK_D': 32}, num_warps=8),
    ],
    key=['dim1', 'dim2'],
)
@triton.jit
def _full_tail_kernel(
    tmp4_ptr,
    in4_ptr,
    in6_ptr,
    out_ptr,
    scale,
    batch,
    heads,
    channels,
    dim1,
    dim2,
    stride_tmp4_b,
    stride_tmp4_h,
    stride_tmp4_c,
    stride_tmp4_t,
    stride_in4_b,
    stride_in4_h,
    stride_in4_t,
    stride_in4_c,
    stride_in6_b,
    stride_in6_h,
    stride_in6_t,
    stride_in6_c,
    stride_out_b,
    stride_out_t,
    stride_out_d,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    t1 = offs_t[:, None]
    d = offs_d[None, :]
    h = d // channels
    c = d - h * channels
    mask = (pid_b < batch) & (t1 < dim1) & (d < dim2)

    in4_ptrs = (
        in4_ptr
        + pid_b * stride_in4_b
        + h * stride_in4_h
        + t1 * stride_in4_t
        + c * stride_in4_c
    )
    acc = tl.load(in4_ptrs, mask=mask, other=0).to(tl.float32) * scale

    t = t1 - 1
    prod_mask = mask & (t1 > 0)
    in6_ptrs = (
        in6_ptr
        + pid_b * stride_in6_b
        + h * stride_in6_h
        + t * stride_in6_t
        + c * stride_in6_c
    )
    tmp4_ptrs = (
        tmp4_ptr
        + pid_b * stride_tmp4_b
        + h * stride_tmp4_h
        + c * stride_tmp4_c
        + t * stride_tmp4_t
    )
    prod = tl.load(in6_ptrs, mask=prod_mask, other=0).to(tl.float32)
    prod = prod * tl.load(tmp4_ptrs, mask=prod_mask, other=0).to(tl.float32)
    tl.store(
        out_ptr + pid_b * stride_out_b + t1 * stride_out_t + d * stride_out_d,
        acc + prod,
        mask=mask,
    )



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 8, 'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_T': 16, 'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_T': 8, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 16, 'BLOCK_C': 64}, num_warps=4),
    ],
    key=['dim1', 'channels'],
)
@triton.jit
def _full_tail_head_kernel(
    tmp4_ptr,
    in4_ptr,
    in6_ptr,
    out_ptr,
    scale,
    batch,
    heads,
    channels,
    dim1,
    stride_tmp4_b,
    stride_tmp4_h,
    stride_tmp4_c,
    stride_tmp4_t,
    stride_in4_b,
    stride_in4_h,
    stride_in4_t,
    stride_in4_c,
    stride_in6_b,
    stride_in6_h,
    stride_in6_t,
    stride_in6_c,
    stride_out_b,
    stride_out_t,
    stride_out_d,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, BLOCK_C)

    t1 = offs_t[:, None]
    c = offs_c[None, :]

    mask = (pid_b < batch) & (pid_h < heads) & (t1 < dim1) & (c < channels)

    in4_ptrs = (
        in4_ptr
        + pid_b * stride_in4_b
        + pid_h * stride_in4_h
        + t1 * stride_in4_t
        + c * stride_in4_c
    )
    acc = tl.load(in4_ptrs, mask=mask, other=0).to(tl.float32) * scale

    t = t1 - 1
    prod_mask = mask & (t1 > 0)
    in6_ptrs = (
        in6_ptr
        + pid_b * stride_in6_b
        + pid_h * stride_in6_h
        + t * stride_in6_t
        + c * stride_in6_c
    )
    tmp4_ptrs = (
        tmp4_ptr
        + pid_b * stride_tmp4_b
        + pid_h * stride_tmp4_h
        + c * stride_tmp4_c
        + t * stride_tmp4_t
    )
    prod = tl.load(in6_ptrs, mask=prod_mask, other=0).to(tl.float32)
    prod = prod * tl.load(tmp4_ptrs, mask=prod_mask, other=0).to(tl.float32)

    out_col = pid_h * channels + c
    out_ptrs = (
        out_ptr
        + pid_b * stride_out_b
        + t1 * stride_out_t
        + out_col * stride_out_d
    )
    tl.store(out_ptrs, acc + prod, mask=mask)



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 8, 'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_T': 16, 'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_T': 32, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_T': 8, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 16, 'BLOCK_C': 64}, num_warps=4),
    ],
    key=['dim1', 'channels'],
)
@triton.jit
def _full_tail_batch1_kernel(
    tmp4_ptr,
    in4_ptr,
    in6_ptr,
    out_ptr,
    scale,
    heads,
    channels,
    dim1,
    stride_tmp4_h,
    stride_tmp4_c,
    stride_tmp4_t,
    stride_in4_h,
    stride_in4_t,
    stride_in4_c,
    stride_in6_h,
    stride_in6_t,
    stride_in6_c,
    stride_out_t,
    stride_out_d,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = tl.arange(0, BLOCK_C)

    t1 = offs_t[:, None]
    c = offs_c[None, :]
    mask = (pid_h < heads) & (t1 < dim1) & (c < channels)

    in4_ptrs = in4_ptr + pid_h * stride_in4_h + t1 * stride_in4_t + c * stride_in4_c
    acc = tl.load(in4_ptrs, mask=mask, other=0).to(tl.float32) * scale

    t = t1 - 1
    prod_mask = mask & (t1 > 0)
    in6_ptrs = in6_ptr + pid_h * stride_in6_h + t * stride_in6_t + c * stride_in6_c
    tmp4_ptrs = tmp4_ptr + pid_h * stride_tmp4_h + c * stride_tmp4_c + t * stride_tmp4_t
    prod = tl.load(in6_ptrs, mask=prod_mask, other=0).to(tl.float32)
    prod = prod * tl.load(tmp4_ptrs, mask=prod_mask, other=0).to(tl.float32)

    out_col = pid_h * channels + c
    out_ptrs = out_ptr + t1 * stride_out_t + out_col * stride_out_d
    tl.store(out_ptrs, acc + prod, mask=mask)


@torch.fx.wrap
def factoratt_dispatch(*args):
    route = args[-1]
    if route == 'transpose_mul':
        tmp_4, in_6, route = args
        batch = tmp_4.shape[0]
        heads = tmp_4.shape[1]
        channels = tmp_4.shape[2]
        tokens = tmp_4.shape[3]
        dmodel = heads * channels
        out = torch.empty((batch, heads, tokens, channels), device=tmp_4.device, dtype=tmp_4.dtype)
        grid = lambda META: (
            triton.cdiv(tokens, META['BLOCK_M']),
            triton.cdiv(dmodel, META['BLOCK_N']),
            batch * heads,
        )
        _transpose_mul_kernel[grid](
            tmp_4,
            in_6,
            out,
            batch,
            heads,
            channels,
            tokens,
            dmodel,
            tmp_4.stride(0),
            tmp_4.stride(1),
            tmp_4.stride(2),
            tmp_4.stride(3),
            in_6.stride(0),
            in_6.stride(1),
            in_6.stride(2),
            in_6.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
        return out
    elif route == 'scale_add_transpose_reshape':
        in_4, scale, padded, dim1, dim2, route = args
        batch = in_4.shape[0]
        heads = in_4.shape[1]
        channels = in_4.shape[3]
        dim1 = int(dim1)
        dim2 = int(dim2)
        out = torch.empty((batch, dim1, dim2), device=in_4.device, dtype=in_4.dtype)
        grid = lambda META: (
            triton.cdiv(dim1, META['BLOCK_T']),
            triton.cdiv(dim2, META['BLOCK_D']),
            batch,
        )
        _scale_add_transpose_reshape_kernel[grid](
            in_4,
            padded,
            out,
            scale,
            batch,
            heads,
            channels,
            dim1,
            dim2,
            in_4.stride(0),
            in_4.stride(1),
            in_4.stride(2),
            in_4.stride(3),
            padded.stride(0),
            padded.stride(1),
            padded.stride(2),
            padded.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
        )
        return out
    elif route == 'full_tail':
        tmp_4, in_4, scale, in_6, dim1, dim2, route = args
        batch = tmp_4.shape[0]
        heads = tmp_4.shape[1]
        channels = tmp_4.shape[2]
        dim1 = int(dim1)
        dim2 = int(dim2)
        out = torch.empty((batch, dim1, dim2), device=in_4.device, dtype=in_4.dtype)
        if batch == 1:
            grid = lambda META: (
                triton.cdiv(dim1, META['BLOCK_T']),
                heads,
            )
            _full_tail_batch1_kernel[grid](
                tmp_4,
                in_4,
                in_6,
                out,
                scale,
                heads,
                channels,
                dim1,
                tmp_4.stride(1),
                tmp_4.stride(2),
                tmp_4.stride(3),
                in_4.stride(1),
                in_4.stride(2),
                in_4.stride(3),
                in_6.stride(1),
                in_6.stride(2),
                in_6.stride(3),
                out.stride(1),
                out.stride(2),
            )
        else:
            grid = lambda META: (
                triton.cdiv(dim1, META['BLOCK_T']),
                heads,
                batch,
            )
            _full_tail_head_kernel[grid](
                tmp_4,
                in_4,
                in_6,
                out,
                scale,
                batch,
                heads,
                channels,
                dim1,
                tmp_4.stride(0),
                tmp_4.stride(1),
                tmp_4.stride(2),
                tmp_4.stride(3),
                in_4.stride(0),
                in_4.stride(1),
                in_4.stride(2),
                in_4.stride(3),
                in_6.stride(0),
                in_6.stride(1),
                in_6.stride(2),
                in_6.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
            )
        return out
    raise RuntimeError(f'Unknown route: {route}')


def shared_replacement_func():
    return factoratt_dispatch