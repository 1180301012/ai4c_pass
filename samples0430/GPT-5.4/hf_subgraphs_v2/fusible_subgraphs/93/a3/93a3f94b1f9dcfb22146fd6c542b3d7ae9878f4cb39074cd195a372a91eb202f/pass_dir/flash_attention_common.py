import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_om, stride_ok,
    Z, H, M, N, D,
    input_dtype_code,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)

    z = pid_zh // H
    h = pid_zh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    q_base = q_ptr + z * stride_qz + h * stride_qh
    k_base = k_ptr + z * stride_kz + h * stride_kh
    v_base = v_ptr + z * stride_vz + h * stride_vh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q = q.to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N, BLOCK_N):
        cur_n = start_n + offs_n
        k_ptrs = k_base + offs_d[:, None] * stride_kk + cur_n[None, :] * stride_kn
        k_mask = (offs_d[:, None] < D) & (cur_n[None, :] < N)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k = k.to(tl.float32)

        qk = tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = v_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = (cur_n[:, None] < N) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        v = v.to(tl.float32)

        acc += tl.dot(p, v)
        m_i = m_ij

    acc = acc / l_i[:, None]

    offs_out_k = h * D + offs_d
    out_base = out_ptr + z * stride_oz
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_out_k[None, :] * stride_ok
    out_mask = (offs_m[:, None] < M) & (offs_out_k[None, :] < H * D) & (offs_d[None, :] < D)

    if input_dtype_code == 0:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    elif input_dtype_code == 1:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'D'],
)
@triton.jit
def _attn_value_matmul_reshape_kernel(
    x_ptr,
    v_ptr,
    out_ptr,
    stride_xz, stride_xh, stride_xm, stride_xn,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_om, stride_ok,
    Z, H, M, N, D,
    output_dtype_code,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hd = tl.program_id(1)
    z = tl.program_id(2)

    h = pid_hd // tl.cdiv(D, BLOCK_D)
    d_blk = pid_hd % tl.cdiv(D, BLOCK_D)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    x_base = x_ptr + z * stride_xz + h * stride_xh
    v_base = v_ptr + z * stride_vz + h * stride_vh

    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N, BLOCK_N):
        cur_n = start_n + offs_n
        x_ptrs = x_base + offs_m[:, None] * stride_xm + cur_n[None, :] * stride_xn
        x_mask = (offs_m[:, None] < M) & (cur_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        x = x.to(tl.float32)

        v_ptrs = v_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (cur_n[:, None] < N) & (offs_d[None, :] < D)
        vv = tl.load(v_ptrs, mask=v_mask, other=0.0)
        vv = vv.to(tl.float32)

        acc += tl.dot(x, vv)

    out_base = out_ptr + z * stride_oz
    offs_out_k = h * D + offs_d
    out_ptrs = out_base + offs_m[:, None] * stride_om + offs_out_k[None, :] * stride_ok
    out_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)

    if output_dtype_code == 0:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)
    elif output_dtype_code == 1:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def flash_attention_qk_softmax_v_reshape(q, k=None, v=None, route=None):
    if isinstance(v, str) and route is None:
        route = v
        v = k
        k = None

    if q.dtype == torch.float16:
        output_dtype_code = 0
    elif q.dtype == torch.bfloat16:
        output_dtype_code = 1
    else:
        output_dtype_code = 2

    if route == 'sdpa_full':
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.transpose(-2, -1),
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
        return attn_out.transpose(1, 2).contiguous().reshape(q.shape[0], q.shape[2], -1).contiguous()

    if route == 'matmul_tail':
        x = q
        Z, H, M, N = x.shape
        D = v.shape[-1]
        out = torch.empty((Z, M, H * D), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 64), H * triton.cdiv(D, 64), Z)
        _attn_value_matmul_reshape_kernel[grid](
            x,
            v,
            out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            Z, H, M, N, D,
            output_dtype_code,
        )
        return out

    Z, H, M, D = q.shape
    N = k.shape[-1]
    out = torch.empty((Z, M, H * D), device=q.device, dtype=q.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    grid = (triton.cdiv(M, BLOCK_M), Z * H)

    _flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        Z, H, M, N, D,
        output_dtype_code,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out