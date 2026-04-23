import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 16, "BLOCK_K": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_K": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_K": 64, "BLOCK_D": 32}, num_warps=8, num_stages=2),
    ],
    key=["S"],
)
@triton.jit
def _fused_attn_kernel(
    a_ptr,
    b_ptr,
    v_ptr,
    out_ptr,
    stride_a_b,
    stride_a_q,
    stride_a_k,
    stride_b_b,
    stride_b_q,
    stride_b_k,
    stride_v_b,
    stride_v_k,
    stride_v_d,
    stride_out_b,
    stride_out_d,
    stride_out_q,
    S,
    D,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_q = offs_q < S
    mask_d = offs_d < D

    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, S, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < S
        qk_mask = mask_q[:, None] & mask_k[None, :]

        a = tl.load(
            a_ptr + pid_b * stride_a_b + offs_q[:, None] * stride_a_q + offs_k[None, :] * stride_a_k,
            mask=qk_mask,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + pid_b * stride_b_b + offs_q[:, None] * stride_b_q + offs_k[None, :] * stride_b_k,
            mask=qk_mask,
            other=0.0,
        ).to(tl.float32)
        logits = a + b
        logits = tl.where(qk_mask, logits, -float("inf"))

        m_ij = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_ptr + pid_b * stride_v_b + offs_k[:, None] * stride_v_k + offs_d[None, :] * stride_v_d,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)

        m_i = m_new
        l_i = l_new

    acc = acc / l_i[:, None]
    out_vals = tl.trans(acc)
    out_offsets = pid_b * stride_out_b + offs_d[:, None] * stride_out_d + offs_q[None, :] * stride_out_q
    tl.store(out_ptr + out_offsets, out_vals, mask=mask_d[:, None] & mask_q[None, :])


def pattern(in_0, in_1, in_2):
    tmp_12 = in_0 + in_1
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_2
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_add_softmax_matmul_transpose(in_0, in_1, in_2):
    bsz = in_0.shape[0]
    s = in_0.shape[1]
    d = in_2.shape[2]
    out = torch.empty((bsz, d, s), device=in_0.device, dtype=in_0.dtype)

    grid = lambda META: (triton.cdiv(s, META["BLOCK_Q"]), triton.cdiv(d, META["BLOCK_D"]), bsz)
    _fused_attn_kernel[grid](
        a_ptr=in_0,
        b_ptr=in_1,
        v_ptr=in_2,
        out_ptr=out,
        stride_a_b=in_0.stride(0),
        stride_a_q=in_0.stride(1),
        stride_a_k=in_0.stride(2),
        stride_b_b=in_1.stride(0),
        stride_b_q=in_1.stride(1),
        stride_b_k=in_1.stride(2),
        stride_v_b=in_2.stride(0),
        stride_v_k=in_2.stride(1),
        stride_v_d=in_2.stride(2),
        stride_out_b=out.stride(0),
        stride_out_d=out.stride(1),
        stride_out_q=out.stride(2),
        S=s,
        D=d,
    )
    return out


def replacement_func():
    return fused_add_softmax_matmul_transpose