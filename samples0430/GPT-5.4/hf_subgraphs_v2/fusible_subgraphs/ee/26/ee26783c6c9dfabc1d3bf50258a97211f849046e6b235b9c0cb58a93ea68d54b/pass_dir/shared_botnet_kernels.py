import torch
import triton
import triton.language as tl


@triton.jit
def _relpos_scores_16_kernel(
    base_ptr,
    q_ptr,
    rel2_ptr,
    w_ptr,
    out_ptr,
    base_s0,
    base_s1,
    base_s2,
    q_s0,
    q_s1,
    q_s2,
    q_s3,
    rel2_s0,
    rel2_s1,
    rel2_s2,
    rel2_s3,
    rel2_s4,
    w_s0,
    w_s1,
    out_s0,
    out_s1,
    out_s2,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    D_MODEL: tl.constexpr,
):
    q_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    hq = q_idx % H
    wq = q_idx // H

    offs_w = tl.arange(0, W)
    rel_idx = offs_w - wq + (W - 1)
    rel_bias_w = tl.zeros([W], dtype=tl.float32)

    for d_start in range(0, D_MODEL, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        q_ptrs = q_ptr + b_idx * q_s0 + hq * q_s1 + wq * q_s2 + offs_d * q_s3
        q_vals = tl.load(q_ptrs, mask=offs_d < D_MODEL, other=0.0).to(tl.float32)

        w_ptrs = w_ptr + offs_d[:, None] * w_s0 + rel_idx[None, :] * w_s1
        w_vals = tl.load(w_ptrs, mask=(offs_d[:, None] < D_MODEL), other=0.0).to(tl.float32)
        rel_bias_w += tl.sum(q_vals[:, None] * w_vals, axis=0)

    offs_k = tl.arange(0, BLOCK_M)
    wk = offs_k // H
    hk = offs_k % H

    rel_bias = tl.zeros([BLOCK_M], dtype=tl.float32)
    for wpos in range(W):
        rel_bias = tl.where(wk == wpos, rel_bias_w[wpos], rel_bias)

    base_ptrs = base_ptr + b_idx * base_s0 + q_idx * base_s1 + offs_k * base_s2
    base_vals = tl.load(base_ptrs).to(tl.float32)

    rel2_ptrs = (
        rel2_ptr
        + b_idx * rel2_s0
        + wq * rel2_s1
        + hq * rel2_s2
        + wk * rel2_s3
        + hk * rel2_s4
    )
    rel2_vals = tl.load(rel2_ptrs).to(tl.float32)

    out_vals = base_vals + rel2_vals + rel_bias
    out_ptrs = out_ptr + b_idx * out_s0 + q_idx * out_s1 + offs_k * out_s2
    tl.store(out_ptrs, out_vals)


@triton.jit
def _relpos_scores_8_kernel(
    base_ptr,
    q_ptr,
    rel2_ptr,
    w_ptr,
    out_ptr,
    base_s0,
    base_s1,
    base_s2,
    q_s0,
    q_s1,
    q_s2,
    q_s3,
    rel2_s0,
    rel2_s1,
    rel2_s2,
    rel2_s3,
    rel2_s4,
    w_s0,
    w_s1,
    out_s0,
    out_s1,
    out_s2,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    D_MODEL: tl.constexpr,
):
    q_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    hq = q_idx % H
    wq = q_idx // H

    offs_w = tl.arange(0, W)
    rel_idx = offs_w - wq + (W - 1)
    rel_bias_w = tl.zeros([W], dtype=tl.float32)

    for d_start in range(0, D_MODEL, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        q_ptrs = q_ptr + b_idx * q_s0 + hq * q_s1 + wq * q_s2 + offs_d * q_s3
        q_vals = tl.load(q_ptrs, mask=offs_d < D_MODEL, other=0.0).to(tl.float32)

        w_ptrs = w_ptr + offs_d[:, None] * w_s0 + rel_idx[None, :] * w_s1
        w_vals = tl.load(w_ptrs, mask=(offs_d[:, None] < D_MODEL), other=0.0).to(tl.float32)
        rel_bias_w += tl.sum(q_vals[:, None] * w_vals, axis=0)

    offs_k = tl.arange(0, BLOCK_M)
    wk = offs_k // H
    hk = offs_k % H

    rel_bias = tl.zeros([BLOCK_M], dtype=tl.float32)
    for wpos in range(W):
        rel_bias = tl.where(wk == wpos, rel_bias_w[wpos], rel_bias)

    base_ptrs = base_ptr + b_idx * base_s0 + q_idx * base_s1 + offs_k * base_s2
    base_vals = tl.load(base_ptrs).to(tl.float32)

    rel2_ptrs = (
        rel2_ptr
        + b_idx * rel2_s0
        + wq * rel2_s1
        + hq * rel2_s2
        + wk * rel2_s3
        + hk * rel2_s4
    )
    rel2_vals = tl.load(rel2_ptrs).to(tl.float32)

    out_vals = base_vals + rel2_vals + rel_bias
    out_ptrs = out_ptr + b_idx * out_s0 + q_idx * out_s1 + offs_k * out_s2
    tl.store(out_ptrs, out_vals)


@triton.jit
def _broadcast_shift_add_16_kernel(
    base_ptr,
    shift_ptr,
    rel2_ptr,
    out_ptr,
    base_s0,
    base_s1,
    base_s2,
    shift_s0,
    shift_s1,
    shift_s2,
    rel2_s0,
    rel2_s1,
    rel2_s2,
    rel2_s3,
    rel2_s4,
    out_s0,
    out_s1,
    out_s2,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    hq = q_idx % H
    wq = q_idx // H

    offs_k = tl.arange(0, BLOCK_M)
    wk = offs_k // H
    hk = offs_k % H

    shift_ptrs = shift_ptr + (b_idx * H + hq) * shift_s0 + wq * shift_s1 + wk * shift_s2
    shift_bcast = tl.load(shift_ptrs).to(tl.float32)

    base_ptrs = base_ptr + b_idx * base_s0 + q_idx * base_s1 + offs_k * base_s2
    base_vals = tl.load(base_ptrs).to(tl.float32)

    rel2_ptrs = (
        rel2_ptr
        + b_idx * rel2_s0
        + wq * rel2_s1
        + hq * rel2_s2
        + wk * rel2_s3
        + hk * rel2_s4
    )
    rel2_vals = tl.load(rel2_ptrs).to(tl.float32)

    out_vals = base_vals + rel2_vals + shift_bcast
    out_ptrs = out_ptr + b_idx * out_s0 + q_idx * out_s1 + offs_k * out_s2
    tl.store(out_ptrs, out_vals)


@triton.jit
def _broadcast_shift_add_8_kernel(
    base_ptr,
    shift_ptr,
    rel2_ptr,
    out_ptr,
    base_s0,
    base_s1,
    base_s2,
    shift_s0,
    shift_s1,
    shift_s2,
    rel2_s0,
    rel2_s1,
    rel2_s2,
    rel2_s3,
    rel2_s4,
    out_s0,
    out_s1,
    out_s2,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    hq = q_idx % H
    wq = q_idx // H

    offs_k = tl.arange(0, BLOCK_M)
    wk = offs_k // H
    hk = offs_k % H

    shift_ptrs = shift_ptr + (b_idx * H + hq) * shift_s0 + wq * shift_s1 + wk * shift_s2
    shift_bcast = tl.load(shift_ptrs).to(tl.float32)

    base_ptrs = base_ptr + b_idx * base_s0 + q_idx * base_s1 + offs_k * base_s2
    base_vals = tl.load(base_ptrs).to(tl.float32)

    rel2_ptrs = (
        rel2_ptr
        + b_idx * rel2_s0
        + wq * rel2_s1
        + hq * rel2_s2
        + wk * rel2_s3
        + hk * rel2_s4
    )
    rel2_vals = tl.load(rel2_ptrs).to(tl.float32)

    out_vals = base_vals + rel2_vals + shift_bcast
    out_ptrs = out_ptr + b_idx * out_s0 + q_idx * out_s1 + offs_k * out_s2
    tl.store(out_ptrs, out_vals)


@triton.jit
def _softmax_matmul_transpose_kernel(
    scores_ptr,
    v_ptr,
    out_ptr,
    scores_s0,
    scores_s1,
    scores_s2,
    v_s0,
    v_s1,
    v_s2,
    out_s0,
    out_s1,
    out_s2,
    N_CTX: tl.constexpr,
    D_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d_block = tl.program_id(0)
    q_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    row_ptr = scores_ptr + b_idx * scores_s0 + q_idx * scores_s1

    m = -float('inf')
    for n_start in range(0, N_CTX, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        score_ptrs = row_ptr + n_offsets * scores_s2
        score = tl.load(score_ptrs, mask=n_offsets < N_CTX, other=-float('inf')).to(tl.float32)
        m = tl.maximum(m, tl.max(score, axis=0))

    denom = 0.0
    for n_start in range(0, N_CTX, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        score_ptrs = row_ptr + n_offsets * scores_s2
        score = tl.load(score_ptrs, mask=n_offsets < N_CTX, other=-float('inf')).to(tl.float32)
        p = tl.exp(score - m)
        denom += tl.sum(p, axis=0)

        v_ptrs = (
            v_ptr
            + b_idx * v_s0
            + n_offsets[:, None] * v_s1
            + d_offsets[None, :] * v_s2
        )
        v_vals = tl.load(
            v_ptrs,
            mask=(n_offsets[:, None] < N_CTX) & (d_offsets[None, :] < D_OUT),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(p[:, None] * v_vals, axis=0)

    out_vals = acc / denom
    out_ptrs = out_ptr + b_idx * out_s0 + d_offsets * out_s1 + q_idx * out_s2
    tl.store(out_ptrs, out_vals, mask=d_offsets < D_OUT)


@torch.fx.wrap
def botnet_dispatch(*args):
    route = args[-1]
    if route == 'relpos16':
        in_0, in_1, in_2, in_3, _ = args
        out = torch.empty_like(in_0)
        grid = (256, in_0.shape[0])
        _relpos_scores_16_kernel[grid](
            in_0,
            in_1,
            in_2,
            in_3,
            out,
            in_0.stride(0),
            in_0.stride(1),
            in_0.stride(2),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
            in_1.stride(3),
            in_2.stride(0),
            in_2.stride(1),
            in_2.stride(2),
            in_2.stride(3),
            in_2.stride(4),
            in_3.stride(0),
            in_3.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_M=256,
            BLOCK_D=32,
            H=16,
            W=16,
            D_MODEL=128,
            num_warps=8,
            num_stages=2,
        )
        return out
    elif route == 'relpos8':
        in_0, in_1, in_2, in_3, _ = args
        out = torch.empty_like(in_0)
        grid = (64, in_0.shape[0])
        _relpos_scores_8_kernel[grid](
            in_0,
            in_1,
            in_2,
            in_3,
            out,
            in_0.stride(0),
            in_0.stride(1),
            in_0.stride(2),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
            in_1.stride(3),
            in_2.stride(0),
            in_2.stride(1),
            in_2.stride(2),
            in_2.stride(3),
            in_2.stride(4),
            in_3.stride(0),
            in_3.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_M=64,
            BLOCK_D=32,
            H=8,
            W=8,
            D_MODEL=128,
            num_warps=4,
            num_stages=2,
        )
        return out
    elif route == 'shiftadd16':
        base, shift, rel2, _ = args
        out = torch.empty_like(base)
        grid = (256, base.shape[0])
        _broadcast_shift_add_16_kernel[grid](
            base,
            shift,
            rel2,
            out,
            base.stride(0),
            base.stride(1),
            base.stride(2),
            shift.stride(0),
            shift.stride(1),
            shift.stride(2),
            rel2.stride(0),
            rel2.stride(1),
            rel2.stride(2),
            rel2.stride(3),
            rel2.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            H=16,
            W=16,
            BLOCK_M=256,
            num_warps=8,
            num_stages=2,
        )
        return out
    elif route == 'shiftadd8':
        base, shift, rel2, _ = args
        out = torch.empty_like(base)
        grid = (64, base.shape[0])
        _broadcast_shift_add_8_kernel[grid](
            base,
            shift,
            rel2,
            out,
            base.stride(0),
            base.stride(1),
            base.stride(2),
            shift.stride(0),
            shift.stride(1),
            shift.stride(2),
            rel2.stride(0),
            rel2.stride(1),
            rel2.stride(2),
            rel2.stride(3),
            rel2.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            H=8,
            W=8,
            BLOCK_M=64,
            num_warps=4,
            num_stages=2,
        )
        return out
    elif route == 'softmax_mm_t_256':
        scores, v, _ = args
        out = torch.empty((scores.shape[0], v.shape[2], scores.shape[1]), device=scores.device, dtype=scores.dtype)
        grid = (triton.cdiv(v.shape[2], 32), scores.shape[1], scores.shape[0])
        _softmax_matmul_transpose_kernel[grid](
            scores,
            v,
            out,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            N_CTX=256,
            D_OUT=v.shape[2],
            BLOCK_N=64,
            BLOCK_D=32,
            num_warps=4,
            num_stages=2,
        )
        return out
    elif route == 'softmax_mm_t_64':
        scores, v, _ = args
        out = torch.empty((scores.shape[0], v.shape[2], scores.shape[1]), device=scores.device, dtype=scores.dtype)
        grid = (triton.cdiv(v.shape[2], 32), scores.shape[1], scores.shape[0])
        _softmax_matmul_transpose_kernel[grid](
            scores,
            v,
            out,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            N_CTX=64,
            D_OUT=v.shape[2],
            BLOCK_N=32,
            BLOCK_D=32,
            num_warps=4,
            num_stages=2,
        )
        return out
    return args[0]


def replacement_func():
    return botnet_dispatch