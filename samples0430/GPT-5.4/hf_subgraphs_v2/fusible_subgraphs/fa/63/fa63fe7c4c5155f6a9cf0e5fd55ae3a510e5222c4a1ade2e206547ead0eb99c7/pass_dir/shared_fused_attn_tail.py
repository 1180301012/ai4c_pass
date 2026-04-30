import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _small_bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    H,
    stride_ab,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_ch,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    batch = pid_bh // H
    head = pid_bh - batch * H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + batch * stride_ab + head * stride_ah
    b_base = b_ptr + batch * stride_bb + head * stride_bh

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_base + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            b_base + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + batch * stride_cb + head * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 256, "BLOCK_D": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_Q": 256, "BLOCK_D": 64}, num_warps=8, num_stages=4),
    ],
    key=["Q", "D"],
)
@triton.jit
def _pack_v_kernel(
    in_ptr,
    out_ptr,
    Q,
    D,
    SIDE,
    D_TILES,
    in_stride_b,
    in_stride_h,
    in_stride_q,
    in_stride_d,
    out_stride_b,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_bd = tl.program_id(2)

    batch = pid_bd // D_TILES
    d_tile = pid_bd - batch * D_TILES
    d_off_base = d_tile * BLOCK_D

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]
    offs_d = d_off_base + tl.arange(0, BLOCK_D)[None, :]

    mask = (offs_q < Q) & (offs_d < D)
    src_ptrs = in_ptr + batch * in_stride_b + pid_h * in_stride_h + (offs_q + 1) * in_stride_q + offs_d * in_stride_d
    vals = tl.load(src_ptrs, mask=mask, other=0)

    row = offs_q // SIDE
    col = offs_q - row * SIDE
    channel = pid_h * D + offs_d
    dst_ptrs = out_ptr + batch * out_stride_b + channel * out_stride_c + row * out_stride_h + col * out_stride_w
    tl.store(dst_ptrs, vals, mask=mask)


def _triton_small_bmm(a, b):
    batch, heads, m, k = a.shape
    n = b.shape[3]
    out = torch.empty((batch, heads, m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
        batch * heads,
    )
    _small_bmm_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        heads,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a.stride(3),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def _pack_v_to_tmp4(in_2):
    batch, heads, q_total, d = in_2.shape
    q_tokens = q_total - 1
    side = int(q_tokens ** 0.5)
    if side * side != q_tokens:
        raise RuntimeError(f"Expected square token grid, got {q_tokens} tokens after cls removal")
    out = torch.empty((batch, heads * d, side, side), device=in_2.device, dtype=in_2.dtype)
    d_tiles = triton.cdiv(d, 64)
    grid = lambda meta: (
        triton.cdiv(q_tokens, meta["BLOCK_Q"]),
        heads,
        batch * triton.cdiv(d, meta["BLOCK_D"]),
    )
    _pack_v_kernel[grid](
        in_2,
        out,
        q_tokens,
        d,
        side,
        triton.cdiv(d, 64),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


@torch.fx.wrap
def fused_matmul_vsplit(in_0, in_1, in_2):
    matmul = _triton_small_bmm(in_1, in_0)
    tmp_1 = in_1[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tmp_4 = _pack_v_to_tmp4(in_2)
    d = in_2.shape[3]
    tmp_6, tmp_7, tmp_8 = torch.functional.split(tmp_4, [2 * d, 3 * d, 3 * d], dim=1)
    return (matmul, tmp_6, tmp_7, tmp_8, tmp_1)