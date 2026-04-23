import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"CHUNK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"CHUNK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"CHUNK_D": 32}, num_warps=8, num_stages=2),
        triton.Config({"CHUNK_D": 64}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _botnet_full_attn_kernel(
    base_ptr,
    q_ptr,
    hbias_ptr,
    rel_ptr,
    v_ptr,
    out_ptr,
    stride_base_b,
    stride_base_q,
    stride_base_k,
    stride_q_b,
    stride_q_y,
    stride_q_x,
    stride_q_d,
    stride_h_b,
    stride_h_qx,
    stride_h_qy,
    stride_h_kx,
    stride_h_ky,
    stride_rel_d,
    stride_rel_r,
    stride_v_b,
    stride_v_k,
    stride_v_d,
    stride_out_b,
    stride_out_d,
    stride_out_q,
    N: tl.constexpr,
    DMODEL: tl.constexpr,
    CHUNK_D: tl.constexpr,
):
    pid_qx = tl.program_id(0)
    pid_b = tl.program_id(1)

    qy = tl.arange(0, N)
    ky = tl.arange(0, N)
    d_all = tl.arange(0, DMODEL)
    q_flat = pid_qx * N + qy

    m_i = tl.full((N,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((N,), dtype=tl.float32)
    acc = tl.zeros((N, DMODEL), dtype=tl.float32)

    for kx in range(0, N):
        rel_col = kx - pid_qx + (N - 1)
        rel_bias = tl.zeros((N,), dtype=tl.float32)
        for d_start in range(0, DMODEL, CHUNK_D):
            d = d_start + tl.arange(0, CHUNK_D)
            q_chunk = tl.load(
                q_ptr + pid_b * stride_q_b + qy[:, None] * stride_q_y + pid_qx * stride_q_x + d[None, :] * stride_q_d,
            ).to(tl.float32)
            rel_chunk = tl.load(
                rel_ptr + d * stride_rel_d + rel_col * stride_rel_r,
            ).to(tl.float32)
            rel_bias += tl.sum(q_chunk * rel_chunk[None, :], axis=1)

        k_flat = kx * N + ky
        logits_base = tl.load(
            base_ptr + pid_b * stride_base_b + q_flat[:, None] * stride_base_q + k_flat[None, :] * stride_base_k,
        ).to(tl.float32)
        logits_h = tl.load(
            hbias_ptr + pid_b * stride_h_b + pid_qx * stride_h_qx + qy[:, None] * stride_h_qy + kx * stride_h_kx + ky[None, :] * stride_h_ky,
        ).to(tl.float32)
        logits = logits_base + logits_h + rel_bias[:, None]

        m_ij = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_ptr + pid_b * stride_v_b + k_flat[:, None] * stride_v_k + d_all[None, :] * stride_v_d,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m_i = m_new

    acc = acc / l_i[:, None]
    out_offsets = pid_b * stride_out_b + d_all[:, None] * stride_out_d + q_flat[None, :] * stride_out_q
    tl.store(out_ptr + out_offsets, tl.trans(acc))


def _botnet_full_attn_impl(in_0, in_1, in_2, in_3, in_4, n):
    b = in_0.shape[0]
    s = n * n
    out = torch.empty((b, in_4.shape[-1], s), device=in_0.device, dtype=in_0.dtype)
    _botnet_full_attn_kernel[(n, b)](
        base_ptr=in_0,
        q_ptr=in_1,
        hbias_ptr=in_2,
        rel_ptr=in_3,
        v_ptr=in_4,
        out_ptr=out,
        stride_base_b=in_0.stride(0),
        stride_base_q=in_0.stride(1),
        stride_base_k=in_0.stride(2),
        stride_q_b=in_1.stride(0),
        stride_q_y=in_1.stride(1),
        stride_q_x=in_1.stride(2),
        stride_q_d=in_1.stride(3),
        stride_h_b=in_2.stride(0),
        stride_h_qx=in_2.stride(1),
        stride_h_qy=in_2.stride(2),
        stride_h_kx=in_2.stride(3),
        stride_h_ky=in_2.stride(4),
        stride_rel_d=in_3.stride(0),
        stride_rel_r=in_3.stride(1),
        stride_v_b=in_4.stride(0),
        stride_v_k=in_4.stride(1),
        stride_v_d=in_4.stride(2),
        stride_out_b=out.stride(0),
        stride_out_d=out.stride(1),
        stride_out_q=out.stride(2),
        N=n,
        DMODEL=in_4.shape[-1],
    )
    return out


@torch.fx.wrap
def botnet_full_attn_dispatch(in_0, in_1, in_2, in_3, in_4, route):
    if route == "botnet_full_attn_16":
        return _botnet_full_attn_impl(in_0, in_1, in_2, in_3, in_4, 16)
    if route == "botnet_full_attn_8":
        return _botnet_full_attn_impl(in_0, in_1, in_2, in_3, in_4, 8)
    raise RuntimeError(f"Unsupported route: {route}")