import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sdpa_B1H4S12(
    V_lin_ptr, Q_ptr, K_ptr, Mask_ptr, Out_ptr,
    B, H,
    ACTUAL_S: tl.constexpr, D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """Fused: value-transpose + SDPA + output-reshape for S=12, D=64, H=4."""
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    s_offs = tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, D)
    s_mask = s_offs < ACTUAL_S
    # Load Q, K: [B, H, ACTUAL_S, D]
    qk_base = b * H * ACTUAL_S * D + h * ACTUAL_S * D
    Q = tl.load(Q_ptr + qk_base + s_offs[:, None] * D + d_offs[None, :], mask=s_mask[:, None], other=0.0)
    K = tl.load(K_ptr + qk_base + s_offs[:, None] * D + d_offs[None, :], mask=s_mask[:, None], other=0.0)
    # Load V from v_lin viewed as [B, ACTUAL_S, H, D]
    v_base = b * ACTUAL_S * H * D + h * D
    V = tl.load(V_lin_ptr + v_base + s_offs[:, None] * (H * D) + d_offs[None, :], mask=s_mask[:, None], other=0.0)
    # Scores = Q @ K.T * scale
    scores = tl.dot(Q, tl.trans(K), allow_tf32=False) * 0.125
    # Load and add mask (broadcast over heads)
    mask_vals = tl.load(Mask_ptr + b * ACTUAL_S * ACTUAL_S + s_offs[:, None] * ACTUAL_S + s_offs[None, :],
                        mask=s_mask[:, None] & s_mask[None, :], other=-3e4)
    scores = scores + mask_vals
    # Mask out padding rows/cols
    pad_mask = (~s_mask)[:, None] | (~s_mask)[None, :]
    scores = tl.where(pad_mask, -3e4, scores)
    # Softmax
    row_max = tl.max(scores, axis=1)[:, None]
    exp_s = tl.exp(scores - row_max)
    attn = exp_s / tl.sum(exp_s, axis=1)[:, None]
    # Output = attn @ V
    out = tl.dot(attn, V, allow_tf32=False)
    # Store to Out: [B, ACTUAL_S, H*D]
    out_base = b * ACTUAL_S * H * D + h * D
    tl.store(Out_ptr + out_base + s_offs[:, None] * (H * D) + d_offs[None, :], out, mask=s_mask[:, None])


@torch.fx.wrap
def _fw_B1H4S12(in_0, in_1, in_2, in_3, in_4, in_5):
    B, H = 1, 4
    device, dtype = in_3.device, in_3.dtype
    w = in_1.to(device=device, dtype=dtype)
    b_bias = in_0.to(device=device, dtype=dtype)
    v_lin = in_3 @ w.transpose(0, 1) + b_bias  # [1, 12, 256]
    S = in_3.shape[1]
    out = torch.empty(B, S, H * 64, dtype=dtype, device=device)
    _fused_sdpa_B1H4S12[(B * H,)](
        v_lin, in_5, in_4, in_2, out, B, H,
        ACTUAL_S=12, D=64, BLOCK_S=16, num_warps=4,
    )
    return out


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    sdpa = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = sdpa.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 12, 256)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return _fw_B1H4S12