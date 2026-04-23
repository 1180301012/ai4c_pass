import torch
import triton
import triton.language as tl

# ============================================================
# Triton kernel: Fused attention - row-wise flash attention
# ============================================================

@triton.jit
def attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qm, stride_qk,
    stride_kb, stride_kn, stride_kk,
    stride_vb, stride_vn, stride_vk,
    stride_ob, stride_om, stride_ok,
    N_q, N_k, D_head: tl.constexpr,
    scale,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    q_ptrs = Q_ptr + pid_bh * stride_qb + pid_m * stride_qm + tl.arange(0, D_head) * stride_qk
    q = tl.load(q_ptrs).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.full([D_head], 0.0, dtype=tl.float32)

    for j_start in range(0, N_k, BLOCK_N):
        j_offsets = j_start + tl.arange(0, BLOCK_N)
        j_mask = j_offsets < N_k

        k_ptrs = K_ptr + pid_bh * stride_kb + j_offsets[:, None] * stride_kn + tl.arange(0, D_head)[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k, axis=1)
        scores *= scale

        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        l_i_new = l_i * alpha
        P = tl.exp(scores - m_new)
        l_ij = tl.sum(P, axis=0)
        l_new = l_i_new + l_ij

        acc = acc * alpha

        v_ptrs = V_ptr + pid_bh * stride_vb + j_offsets[:, None] * stride_vn + tl.arange(0, D_head)[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0).to(tl.float32)

        P_scaled = P / l_new
        acc += tl.sum(P_scaled[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    acc = acc / l_i

    o_ptrs = Out_ptr + pid_bh * stride_ob + pid_m * stride_om + tl.arange(0, D_head) * stride_ok
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty))


def flash_attn_impl(q, k, v, scale):
    batch = q.shape[0]
    heads = q.shape[1]
    N_q = q.shape[2]
    d_k = q.shape[3]
    N_k = k.shape[3]
    d_v = v.shape[3]

    scale_val = 1.0 / scale

    k_t = k.transpose(-1, -2).contiguous()
    q = q.contiguous()
    k_t = k_t.contiguous()
    v = v.contiguous()

    q_3d = q.reshape(batch * heads, N_q, d_k)
    k_3d = k_t.reshape(batch * heads, N_k, d_k)
    v_3d = v.reshape(batch * heads, N_k, d_v)

    out_3d = torch.empty((batch * heads, N_q, d_v), dtype=q.dtype, device=q.device)

    BLOCK_N = 64
    grid = (N_q, batch * heads)

    attn_fwd_kernel[grid](
        q_3d, k_3d, v_3d, out_3d,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
        v_3d.stride(0), v_3d.stride(1), v_3d.stride(2),
        out_3d.stride(0), out_3d.stride(1), out_3d.stride(2),
        N_q, N_k, d_k,
        scale_val,
        BLOCK_N=BLOCK_N,
    )

    out_4d = out_3d.reshape(batch, heads, N_q, d_v)
    out_perm = out_4d.permute(0, 2, 1, 3).contiguous()
    return out_perm


@torch.fx.wrap
def flash_attention_dispatch(q, k, v, route):
    if route == "route_5_656854249492381_p0":
        return flash_attn_impl(q, k, v, 5.656854249492381)
    elif route == "route_8_p0":
        return flash_attn_impl(q, k, v, 8.0)
    elif route == "route_6_p0":
        return flash_attn_impl(q, k, v, 6.0)
    elif route == "route_6_928203230275509_p0_1":
        return flash_attn_impl(q, k, v, 6.928203230275509)
    elif route == "route_6_p0_1":
        return flash_attn_impl(q, k, v, 6.0)
    elif route == "route_8_p0_1":
        return flash_attn_impl(q, k, v, 8.0)
    elif route == "route_5_656854249492381_p0_1":
        return flash_attn_impl(q, k, v, 5.656854249492381)
    else:
        raise ValueError(f"Unknown route: {route}")