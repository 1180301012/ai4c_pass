import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Pattern to match
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


# ─────────────────────────────────────────────────────────────────────────────
# Fused Triton kernel: one program per (b, h, w) row
#   - Loads key matrix K[b,:,h,:] and query vector q[b,:,h,w]
#   - Computes einsum result via dot-product
#   - Fuses cat([in0_row, einsum_row]) + softmax in one pass
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_J': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_J': 64}, num_warps=8),
    ],
    key=['C', 'J'],
)
@triton.jit
def _fused_einsum_cat_softmax(
    query_ptr, key_ptr, in0_ptr,
    out_full_ptr, out_slice_ptr,
    B, C, H, W, J,
    # query strides [B, C, H, W]
    stride_qb, stride_qc, stride_qh, stride_qw,
    # key strides   [B, C, H, J]
    stride_kb, stride_kc, stride_kh, stride_kj,
    # in0 strides   [B, H, W, J]
    stride_i0b, stride_i0h, stride_i0w, stride_i0j,
    BLOCK_C: tl.constexpr,
    BLOCK_J: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    # ---------- decode (b, h, w) from linear pid ----------
    pid    = tl.program_id(0)
    w_id   = pid % W
    h_id   = (pid // W) % H
    b_id   = pid // (H * W)

    c_idx  = tl.arange(0, BLOCK_C)   # [C=64]
    j_idx  = tl.arange(0, BLOCK_J)   # [J=64]

    # ---------- load query vector q[b, c, h, w] ----------
    q_base = b_id * stride_qb + h_id * stride_qh + w_id * stride_qw
    q = tl.load(query_ptr + q_base + c_idx * stride_qc).to(tl.float32)   # [C]

    # ---------- load key matrix k[b, c, h, j] ----------
    k_base = b_id * stride_kb + h_id * stride_kh
    k = tl.load(
        key_ptr + k_base
        + c_idx[:, None] * stride_kc
        + j_idx[None, :] * stride_kj
    ).to(tl.float32)                                                       # [C, J]

    # ---------- einsum: sum_c q[c] * k[c, j] ----------
    einsum_row = tl.sum(q[:, None] * k, axis=0)                           # [J]

    # ---------- load in_0 row: in_0[b, h, w, j] ----------
    in0_base = b_id * stride_i0b + h_id * stride_i0h + w_id * stride_i0w
    in0_row  = tl.load(in0_ptr + in0_base + j_idx * stride_i0j).to(tl.float32)  # [J]

    # ---------- online softmax over 128-element [in0 || einsum] ----------
    # max for numerical stability (handles -inf in in0_row gracefully)
    max_in0  = tl.max(in0_row, axis=0)
    max_ein  = tl.max(einsum_row, axis=0)
    max_val  = tl.where(max_in0 > max_ein, max_in0, max_ein)

    exp_in0  = tl.exp(in0_row    - max_val)
    exp_ein  = tl.exp(einsum_row - max_val)

    denom    = tl.sum(exp_in0, axis=0) + tl.sum(exp_ein, axis=0)

    soft_in0 = exp_in0 / denom
    soft_ein = exp_ein  / denom

    # ---------- store tmp_3: [B, H, W, 128] ----------
    out_base = (b_id * H * W + h_id * W + w_id) * 128
    tl.store(out_full_ptr + out_base          + j_idx, soft_in0.to(OUTPUT_DTYPE))
    tl.store(out_full_ptr + out_base + 64     + j_idx, soft_ein.to(OUTPUT_DTYPE))

    # ---------- store tmp_4: [B, H, W, J] (= first 64 of tmp_3) ----------
    out4_base = (b_id * H * W + h_id * W + w_id) * J
    tl.store(out_slice_ptr + out4_base + j_idx, soft_in0.to(OUTPUT_DTYPE))


# dtype → triton dtype lookup (module-level, evaluated once)
_DTYPE_MAP = {
    torch.float32:  tl.float32,
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@torch.fx.wrap
def fused_ccnet_forward(in_0, in_1, in_2):
    """
    in_0  : [B, H, W, J]   – energy_H_1 (typically all -inf)
    in_1  : [B, C, H, J]   – key
    in_2  : [B, C, H, W]   – query
    returns: (tmp_3 [B,H,W,128], tmp_4 [B,H,W,64])
    """
    B, C, H, W = in_2.shape
    J           = in_1.shape[-1]
    dtype       = in_0.dtype

    out_full  = torch.empty((B, H, W, 128), dtype=dtype, device=in_0.device)
    out_slice = torch.empty((B, H, W, J),   dtype=dtype, device=in_0.device)

    grid = (B * H * W,)

    _fused_einsum_cat_softmax[grid](
        in_2, in_1, in_0,
        out_full, out_slice,
        B, C, H, W, J,
        # query strides
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        # key strides
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        # in_0 strides
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        OUTPUT_DTYPE=_DTYPE_MAP[dtype],
    )

    return (out_full, out_slice)


# ─────────────────────────────────────────────────────────────────────────────
# Pass interface
# ─────────────────────────────────────────────────────────────────────────────
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_ccnet_forward