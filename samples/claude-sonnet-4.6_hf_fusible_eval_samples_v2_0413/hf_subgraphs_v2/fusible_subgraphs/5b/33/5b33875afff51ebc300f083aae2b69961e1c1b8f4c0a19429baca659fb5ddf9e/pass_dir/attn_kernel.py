import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attn_triton_kernel(
    scores_ptr,
    mask_ptr,
    values_ptr,
    output_ptr,
    B, H, S, D,
    scale,
    str_s_b, str_s_h, str_s_q, str_s_k,
    str_m_k,
    str_v_b, str_v_h, str_v_k, str_v_d,
    str_o_b, str_o_q, str_o_h,
    IS_BF16: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused: scale + add_mask + softmax + weighted_matmul + permute(0,2,1,3).
    One program block handles one (b, h, q) query row.
    Output stored at [b, q, h, :] i.e. permuted [B, S, H, D].
    """
    pid = tl.program_id(0)
    q = pid % S
    bh = pid // S
    b = bh // H
    h = bh % H

    k_offs = tl.arange(0, BLOCK_S)
    k_mask = k_offs < S

    # Load scores row, promote to fp32
    scores = tl.load(
        scores_ptr + b * str_s_b + h * str_s_h + q * str_s_q + k_offs,
        mask=k_mask, other=-1e4
    ).to(tl.float32)

    # Load attention mask (broadcast from [1,1,1,S])
    amask = tl.load(
        mask_ptr + k_offs * str_m_k,
        mask=k_mask, other=0.0
    ).to(tl.float32)

    # Scale + add mask
    scores = scores / scale + amask

    # Numerically stable softmax
    scores_max = tl.max(scores, axis=0)
    scores = tl.exp(scores - scores_max)
    scores = tl.where(k_mask, scores, 0.0)
    scores = scores / tl.sum(scores, axis=0)   # [BLOCK_S]

    # Load the full [BLOCK_S, BLOCK_D] value block
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    v_offs = k_offs[:, None] * str_v_k + d_offs[None, :] * str_v_d
    v_mask = k_mask[:, None] & d_mask[None, :]
    V = tl.load(
        values_ptr + b * str_v_b + h * str_v_h + v_offs,
        mask=v_mask, other=0.0
    ).to(tl.float32)  # [BLOCK_S, BLOCK_D]

    # Weighted sum: output[d] = sum_k scores[k] * V[k,d]
    result = tl.sum(scores[:, None] * V, axis=0)  # [BLOCK_D]

    # Store to output[b, q, h, d]  (permuted layout [B,S,H,D])
    out_ptr = output_ptr + b * str_o_b + q * str_o_q + h * str_o_h + d_offs
    if IS_BF16:
        tl.store(out_ptr, result.to(tl.bfloat16), mask=d_mask)
    else:
        tl.store(out_ptr, result.to(tl.float16), mask=d_mask)


@triton.jit
def _copy_triton_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    """Flat memcpy — used to implement reshape without torch.reshape."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask, other=0), mask=mask)


def _next_pow2(n):
    """Return smallest power of 2 >= n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


@triton.jit
def _weighted_matmul_kernel(
    weights_ptr, values_ptr, output_ptr,
    B, H, S, D,
    str_w_b, str_w_h, str_w_q, str_w_k,
    str_v_b, str_v_h, str_v_k, str_v_d,
    str_o_b, str_o_q, str_o_h,
    IS_BF16: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """output[b,q,h,d] = sum_k weights[b,h,q,k] * values[b,h,k,d]  (permuted store)"""
    pid = tl.program_id(0)
    q = pid % S
    bh = pid // S
    b = bh // H
    h = bh % H

    k_offs = tl.arange(0, BLOCK_S)
    k_mask = k_offs < S

    w = tl.load(weights_ptr + b * str_w_b + h * str_w_h + q * str_w_q + k_offs,
                mask=k_mask, other=0.0).to(tl.float32)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    v_offs = k_offs[:, None] * str_v_k + d_offs[None, :] * str_v_d
    v_mask = k_mask[:, None] & d_mask[None, :]
    V = tl.load(values_ptr + b * str_v_b + h * str_v_h + v_offs,
                mask=v_mask, other=0.0).to(tl.float32)

    result = tl.sum(w[:, None] * V, axis=0)

    out_ptr = output_ptr + b * str_o_b + q * str_o_q + h * str_o_h + d_offs
    if IS_BF16:
        tl.store(out_ptr, result.to(tl.bfloat16), mask=d_mask)
    else:
        tl.store(out_ptr, result.to(tl.float16), mask=d_mask)


@torch.fx.wrap
def fused_matmul_permute(attn_weights, values):
    """Fused matmul + permute(0,2,1,3) + contiguous — single pass for all graphs."""
    B, H, S, _ = attn_weights.shape
    D = values.shape[-1]
    IS_BF16 = (attn_weights.dtype == torch.bfloat16)
    output = torch.empty((B, S, H, D), dtype=attn_weights.dtype, device=attn_weights.device)
    BLOCK_S = _next_pow2(S)
    BLOCK_D = _next_pow2(D)
    _weighted_matmul_kernel[(B * H * S,)](
        attn_weights, values, output,
        B, H, S, D,
        attn_weights.stride(0), attn_weights.stride(1),
        attn_weights.stride(2), attn_weights.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        IS_BF16=IS_BF16, BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        num_warps=2,
    )
    return output


@torch.fx.wrap
def fused_dispatch(scores, in1, mask, values, route):
    """
    Dispatch wrapper shared by all routing passes.
    Computes the fused attention chain and the independent reshape.

    route values:
      "scale8_6x64"       -> scale=8.0,               reshape in1 to [1,S,6,64]
      "scale8_2x64"       -> scale=8.0,               reshape in1 to [1,S,2,64]
      "scale_sqrt8_2x8"   -> scale=2.8284271247461903, reshape in1 to [1,S,2,8]
    """
    B, H, S, _ = scores.shape
    D = values.shape[-1]
    IS_BF16 = (scores.dtype == torch.bfloat16)

    # Decode route into (scale_val, reshape_last_two_dims)
    if route == "scale8_6x64":
        scale_val = 8.0
        r2, r3 = 6, 64
    elif route == "scale8_2x64":
        scale_val = 8.0
        r2, r3 = 2, 64
    elif route == "scale_sqrt8_2x8":
        scale_val = 2.8284271247461903
        r2, r3 = 2, 8
    else:
        # Fallback — should never be reached
        scale_val = 8.0
        r2, r3 = 6, 64

    # ---- Output 1: fused attention [B, S, H, D] ----
    attn_out = torch.empty((B, S, H, D), dtype=scores.dtype, device=scores.device)

    BLOCK_S = _next_pow2(S)
    BLOCK_D = _next_pow2(D)
    grid_attn = (B * H * S,)

    _fused_attn_triton_kernel[grid_attn](
        scores, mask, values, attn_out,
        B, H, S, D, scale_val,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        mask.stride(-1),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
        IS_BF16=IS_BF16,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=2,
    )

    # ---- Output 2: reshape(in1, [1, -1, r2, r3]) via flat copy ----
    # -1 resolves to S because in1.numel() == 1 * S * r2 * r3 for all our graphs
    reshape_out = torch.empty((1, S, r2, r3), dtype=in1.dtype, device=in1.device)
    n_elem = in1.numel()
    COPY_BLOCK = 1024
    grid_copy = ((n_elem + COPY_BLOCK - 1) // COPY_BLOCK,)
    _copy_triton_kernel[grid_copy](in1, reshape_out, n_elem, BLOCK=COPY_BLOCK)

    return (attn_out, reshape_out)


@torch.fx.wrap
def fused_dispatch_simple(scores, mask, values, route):
    """
    Single-output dispatch: returns only the fused attention result [B,S,H,D].
    The independent reshape(in_1) remains in the graph untouched.

    route values:
      "scale8"      -> scale=8.0
      "scale_sqrt8" -> scale=2.8284271247461903
    """
    B, H, S, _ = scores.shape
    D = values.shape[-1]
    IS_BF16 = (scores.dtype == torch.bfloat16)

    if route == "scale8":
        scale_val = 8.0
    elif route == "scale_sqrt8":
        scale_val = 2.8284271247461903
    else:
        scale_val = 8.0

    attn_out = torch.empty((B, S, H, D), dtype=scores.dtype, device=scores.device)

    BLOCK_S = _next_pow2(S)
    BLOCK_D = _next_pow2(D)

    _fused_attn_triton_kernel[(B * H * S,)](
        scores, mask, values, attn_out,
        B, H, S, D, scale_val,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        mask.stride(-1),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
        IS_BF16=IS_BF16,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=2,
    )

    return attn_out