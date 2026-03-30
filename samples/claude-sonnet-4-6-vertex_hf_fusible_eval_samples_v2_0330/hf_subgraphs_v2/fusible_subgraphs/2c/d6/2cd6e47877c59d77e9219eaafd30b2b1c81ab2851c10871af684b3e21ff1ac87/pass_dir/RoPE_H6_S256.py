import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['S', 'Sp1'],
)
@triton.jit
def _rope_q_full_6_256(
    q_cls_ptr,  # in_2: [1, H, 1, D]
    q_ptr,      # in_3: [1, H, S, D]
    cos_ptr,    # in_1: [S, D]
    sin_ptr,    # in_5: [S, D]
    out_ptr,    # output: [1, H, S+1, D]
    H, S, Sp1,
    D: tl.constexpr,
):
    # pid indexes over H * Sp1 rows of the output
    pid = tl.program_id(0)
    h = pid // Sp1
    s_out = pid % Sp1

    d = tl.arange(0, D)
    out_off = (h * Sp1 + s_out) * D

    is_cls = (s_out == 0)
    # Safe s index for q and embeddings: clamp to 0 when is_cls
    s = tl.where(is_cls, 0, s_out - 1)

    # Load q class token (for is_cls branch)
    cls_off = h * D
    cls_val = tl.load(q_cls_ptr + cls_off + d)

    # Load q and compute RoPE (for non-cls branch)
    q_off = (h * S + s) * D
    q = tl.load(q_ptr + q_off + d)
    qp = tl.load(q_ptr + q_off + (d ^ 1))
    c = tl.load(cos_ptr + s * D + d)
    si = tl.load(sin_ptr + s * D + d)
    even = (d & 1) == 0
    rope_val = tl.where(even, q * c - qp * si, q * c + qp * si)

    out = tl.where(is_cls, cls_val, rope_val)
    tl.store(out_ptr + out_off + d, out)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['S', 'Sp1'],
)
@triton.jit
def _rope_k_full_6_256(
    k_ptr,    # in_4: [1, H, S+1, D]
    in0_ptr,  # in_0: [S, 2*D]
    out_ptr,  # output: [1, H, S+1, D]
    H, S, Sp1,
    D: tl.constexpr, D2: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid // Sp1
    s_out = pid % Sp1

    d = tl.arange(0, D)
    out_off = (h * Sp1 + s_out) * D

    is_cls = (s_out == 0)
    # Load k from in_4 at position s_out (covers both cls and body)
    k_off = (h * Sp1 + s_out) * D
    k = tl.load(k_ptr + k_off + d)
    kp = tl.load(k_ptr + k_off + (d ^ 1))

    # in_0 index: s = s_out - 1, clamped to 0 for cls
    s = tl.where(is_cls, 0, s_out - 1)
    i0_off = s * D2
    ck = tl.load(in0_ptr + i0_off + d)
    sk = tl.load(in0_ptr + i0_off + D + d)

    even = (d & 1) == 0
    rope_val = tl.where(even, k * sk - kp * ck, k * sk + kp * ck)

    # cls position: copy k unchanged; body: apply RoPE
    out = tl.where(is_cls, k, rope_val)
    tl.store(out_ptr + out_off + d, out)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[Ellipsis, slice(1, None, 2)]
    tmp_3 = -tmp_2
    tmp_4 = in_3[Ellipsis, slice(None, None, 2)]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, 6, 256, 64))
    tmp_7 = tmp_6 * in_5
    tmp_8 = tmp_1 + tmp_7
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(in_6)
    tmp_11 = in_4[slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None)]
    tmp_12 = in_4[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tensor_split = in_0.tensor_split(2, -1)
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    tmp_16 = tmp_12 * tmp_15
    tmp_17 = tmp_12[Ellipsis, slice(1, None, 2)]
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[Ellipsis, slice(None, None, 2)]
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape((1, 6, 256, 64))
    tmp_22 = tmp_21 * tmp_14
    tmp_23 = tmp_16 + tmp_22
    tmp_24 = torch.cat([tmp_11, tmp_23], dim=2)
    tmp_25 = tmp_24.type_as(in_6)
    return (tmp_25, tmp_10)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@torch.fx.wrap
def _rope_fused_6_256_impl(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    H = in_3.shape[1]
    S = in_3.shape[2]
    D = 64
    Sp1 = S + 1

    q_out = torch.empty((1, H, Sp1, D), dtype=in_3.dtype, device=in_3.device)
    k_out = torch.empty((1, H, Sp1, D), dtype=in_4.dtype, device=in_4.device)

    grid = (H * Sp1,)
    _rope_q_full_6_256[grid](in_2, in_3, in_1, in_5, q_out, H, S, Sp1, D=64)
    _rope_k_full_6_256[grid](in_4, in_0, k_out, H, S, Sp1, D=64, D2=128)

    return k_out.to(in_6.dtype), q_out.to(in_6.dtype)


# Non-wrapped so FX traces into it, producing 2 getitem nodes (matching pattern's 2 outputs)
def _rope_fused_6_256(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    result = _rope_fused_6_256_impl(in_0, in_1, in_2, in_3, in_4, in_5, in_6)
    return result[0], result[1]


def replacement_func():
    return _rope_fused_6_256