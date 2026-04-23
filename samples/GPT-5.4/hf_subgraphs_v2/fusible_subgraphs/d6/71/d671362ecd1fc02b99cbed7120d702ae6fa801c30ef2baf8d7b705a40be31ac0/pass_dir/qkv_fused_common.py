import torch
import triton
import triton.language as tl


def _dtype_size(dtype: torch.dtype) -> int:
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['S', 'DOUT'],
)
@triton.jit
def qkv_scatter_kernel(
    src_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    B,
    S,
    H,
    DQ,
    DK,
    DV,
    DOUT,
    stride_src_b,
    stride_src_s,
    stride_src_d,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_d,
    stride_k_s,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_v_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    s0 = pid_s * BLOCK_M
    s = s0 + tl.arange(0, BLOCK_M)
    n0 = tl.arange(0, BLOCK_N)

    d_q_base = h * DQ + n0
    q_mask = (s[:, None] < S) & (n0[None, :] < DQ)
    q_src_offsets = b * stride_src_b + s[:, None] * stride_src_s + d_q_base[None, :] * stride_src_d
    q_vals = tl.load(src_ptr + q_src_offsets, mask=q_mask, other=0.0)
    q_dst_offsets = (
        b * stride_q_b
        + h * stride_q_h
        + s[:, None] * stride_q_s
        + n0[None, :] * stride_q_d
    )
    tl.store(q_ptr + q_dst_offsets, q_vals, mask=q_mask)

    d_k_base = H * DQ + h * DK + n0
    k_mask = (s[:, None] < S) & (n0[None, :] < DK)
    k_src_offsets = b * stride_src_b + s[:, None] * stride_src_s + d_k_base[None, :] * stride_src_d
    k_vals = tl.load(src_ptr + k_src_offsets, mask=k_mask, other=0.0)
    k_dst_offsets = (
        b * stride_k_b
        + h * stride_k_h
        + n0[:, None] * stride_k_d
        + s[None, :] * stride_k_s
    )
    tl.store(k_ptr + k_dst_offsets, tl.trans(k_vals), mask=tl.trans(k_mask))

    dv0 = 0
    while dv0 < DV:
        dv = dv0 + n0
        v_mask = (s[:, None] < S) & (dv[None, :] < DV)
        d_v_base = H * DQ + H * DK + h * DV + dv
        v_src_offsets = b * stride_src_b + s[:, None] * stride_src_s + d_v_base[None, :] * stride_src_d
        v_vals = tl.load(src_ptr + v_src_offsets, mask=v_mask, other=0.0)
        v_dst_offsets = (
            b * stride_v_b
            + h * stride_v_h
            + s[:, None] * stride_v_s
            + dv[None, :] * stride_v_d
        )
        tl.store(v_ptr + v_dst_offsets, v_vals, mask=v_mask)
        dv0 += BLOCK_N


@torch.fx.wrap
def efficientformer_qkv_reorg(linear_out, ab_cuda, batch_size: int):
    B = batch_size
    S = 49
    H = 8
    DQ = 32
    DK = 32
    DV = 128
    DOUT = DQ + DK + DV

    q = torch.empty((B, H, S, DQ), device=linear_out.device, dtype=linear_out.dtype)
    k = torch.empty((B, H, DK, S), device=linear_out.device, dtype=linear_out.dtype)
    v = torch.empty((B, H, S, DV), device=linear_out.device, dtype=linear_out.dtype)

    src = linear_out.reshape(B, S, DOUT * H)

    grid = lambda meta: (B * H, triton.cdiv(S, meta['BLOCK_M']))
    qkv_scatter_kernel[grid](
        src,
        q,
        k,
        v,
        B,
        S,
        H,
        DQ,
        DK,
        DV,
        DOUT,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
    )
    return q, ab_cuda, k, v


def _route_1(in_0, in_1, in_2, in_3, route):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    ab_cuda = in_0.to(device='cuda')
    return efficientformer_qkv_reorg(linear, ab_cuda, 1)


def _route_4(in_0, in_1, in_2, in_3, route):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    ab_cuda = in_0.to(device='cuda')
    return efficientformer_qkv_reorg(linear, ab_cuda, 4)


def _route_8(in_0, in_1, in_2, in_3, route):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    ab_cuda = in_0.to(device='cuda')
    return efficientformer_qkv_reorg(linear, ab_cuda, 8)


def _route_128(in_0, in_1, in_2, in_3, route):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    ab_cuda = in_0.to(device='cuda')
    return efficientformer_qkv_reorg(linear, ab_cuda, 128)


def _route_256(in_0, in_1, in_2, in_3, route):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    ab_cuda = in_0.to(device='cuda')
    return efficientformer_qkv_reorg(linear, ab_cuda, 256)


def _route_512(in_0, in_1, in_2, in_3, route):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    ab_cuda = in_0.to(device='cuda')
    return efficientformer_qkv_reorg(linear, ab_cuda, 512)


@torch.fx.wrap
def dispatch_qkv_replacement(in_0, in_1, in_2, in_3, route: str):
    if route == 'b1':
        return _route_1(in_0, in_1, in_2, in_3, route)
    if route == 'b4':
        return _route_4(in_0, in_1, in_2, in_3, route)
    if route == 'b8':
        return _route_8(in_0, in_1, in_2, in_3, route)
    if route == 'b128':
        return _route_128(in_0, in_1, in_2, in_3, route)
    if route == 'b256':
        return _route_256(in_0, in_1, in_2, in_3, route)
    if route == 'b512':
        return _route_512(in_0, in_1, in_2, in_3, route)
    raise ValueError(f'Unknown route: {route}')