import torch
import triton
import triton.language as tl

# ===================== Pattern matching =====================
def pattern(in_0, in_1, in_2, in_3, in_4):
    linear = torch.nn.functional.linear(in_4, in_1, None)
    tmp_3 = linear.view(-1, 12)
    tmp_4 = in_0.view(-1)
    tmp_5 = tmp_3[tmp_4]
    tmp_6 = tmp_5.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 64, 12, 64, 64)
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 12, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim = -1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return tmp_22

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

# ===================== Triton Kernels =====================

@triton.jit
def linear_kernel(
    x_ptr, w_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        x_mask = m_mask[:, None] & k_mask[None, :]
        x = tl.load(x_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk,
                     mask=x_mask, other=0.0)

        w_mask = k_mask[:, None] & (tl.arange(0, BLOCK_N)[None, :] < N)
        w = tl.load(w_ptr + k_offsets[:, None] * stride_wk + tl.arange(0, BLOCK_N)[None, :] * stride_wn,
                     mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    out_mask = m_mask[:, None] & (tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(out_ptr + m_offsets[:, None] * stride_om + tl.arange(0, BLOCK_N)[None, :] * stride_on,
             acc, mask=out_mask)


@triton.jit
def fused_bias_add_softmax_kernel(
    linear_ptr, idx_ptr, scores_ptr, mask_ptr, out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    total_rows,
    stride_lm, stride_lh,
    stride_idx,
    stride_sb, stride_sh, stride_sr, stride_sc,
    stride_mb, stride_mr, stride_mc,
    stride_ob, stride_oh, stride_or, stride_oc,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)

    if row_idx >= total_rows:
        return

    b = row_idx // (H * W)
    h = (row_idx // W) % H
    r = row_idx % W

    row_max = tl.zeros([BLOCK_N], dtype=tl.float32) - float('inf')
    row_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Find global max across all columns
    for c_start in range(0, W, BLOCK_N):
        c_offsets = c_start + tl.arange(0, BLOCK_N)
        c_mask = c_offsets < W

        # Load index for each column position
        idx_val = tl.load(idx_ptr + r * W + c_offsets, mask=c_mask, other=0)

        # Compute bias: 16 * sigmoid(linear[idx, h])
        # Load linear output row for each index value
        linear_row = tl.load(linear_ptr + idx_val * stride_lm + h * stride_lh, mask=c_mask, other=0.0)
        bias = 16.0 * tl.sigmoid(linear_row.to(tl.float32))

        # Load attention score
        score = tl.load(scores_ptr + b * stride_sb + h * stride_sh + r * stride_sr + c_offsets * stride_sc,
                        mask=c_mask, other=0.0).to(tl.float32)

        # Load mask values
        mask_val = tl.load(mask_ptr + b * stride_mb + r * stride_mr + c_offsets * stride_mc,
                          mask=c_mask, other=0.0).to(tl.float32)

        # Total value = bias + score + mask
        val = bias + score + mask_val

        # Update row max (for softmax numerical stability)
        row_max = tl.maximum(row_max, tl.where(c_mask, val, float('-inf')))

    # Subtract max and compute exp + sum
    for c_start in range(0, W, BLOCK_N):
        c_offsets = c_start + tl.arange(0, BLOCK_N)
        c_mask = c_offsets < W

        idx_val = tl.load(idx_ptr + r * W + c_offsets, mask=c_mask, other=0)
        linear_row = tl.load(linear_ptr + idx_val * stride_lm + h * stride_lh, mask=c_mask, other=0.0)
        bias = 16.0 * tl.sigmoid(linear_row.to(tl.float32))

        score = tl.load(scores_ptr + b * stride_sb + h * stride_sh + r * stride_sr + c_offsets * stride_sc,
                        mask=c_mask, other=0.0).to(tl.float32)

        mask_val = tl.load(mask_ptr + b * stride_mb + r * stride_mr + c_offsets * stride_mc,
                          mask=c_mask, other=0.0).to(tl.float32)

        val = bias + score + mask_val

        exp_val = tl.where(c_mask, tl.exp(val - row_max), 0.0)
        row_sum += exp_val

    # Normalize by sum and store
    for c_start in range(0, W, BLOCK_N):
        c_offsets = c_start + tl.arange(0, BLOCK_N)
        c_mask = c_offsets < W

        idx_val = tl.load(idx_ptr + r * W + c_offsets, mask=c_mask, other=0)
        linear_row = tl.load(linear_ptr + idx_val * stride_lm + h * stride_lh, mask=c_mask, other=0.0)
        bias = 16.0 * tl.sigmoid(linear_row.to(tl.float32))

        score = tl.load(scores_ptr + b * stride_sb + h * stride_sh + r * stride_sr + c_offsets * stride_sc,
                        mask=c_mask, other=0.0).to(tl.float32)

        mask_val = tl.load(mask_ptr + b * stride_mb + r * stride_mr + c_offsets * stride_mc,
                          mask=c_mask, other=0.0).to(tl.float32)

        val = bias + score + mask_val

        result = tl.where(c_mask, tl.exp(val - row_max) / row_sum, 0.0)

        tl.store(out_ptr + b * stride_ob + h * stride_oh + r * stride_or + c_offsets * stride_oc,
                 result, mask=c_mask)


# ===================== Kernel Wrappers =====================

@torch.fx.wrap
def triton_linear(x, weight):
    M = x.shape[0]
    K = x.shape[1]
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_K = 64
    BLOCK_N = max(16, (N + 15) // 16 * 16)

    grid = ((M + BLOCK_M - 1) // BLOCK_M,)

    linear_kernel[grid](
        x, weight, out,
        M, K, N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    return out


@torch.fx.wrap
def triton_fused_bias_add_softmax(linear_result, rel_pos_idx, attn_scores, attn_mask):
    # Dimensions
    idx_flat = rel_pos_idx.view(-1)  # [W*W] = [4096]
    W = 64  # Window size (fixed for all variants)
    H = linear_result.shape[1]  # Number of heads (12 or 24)

    B = attn_scores.shape[0]  # Batch size for scores
    total_rows = B * H * W

    out = torch.empty_like(attn_scores)  # [B, H, W, W]

    BLOCK_N = 64  # = W, process entire row at once

    grid = (total_rows,)

    fused_bias_add_softmax_kernel[grid](
        linear_result, idx_flat, attn_scores, attn_mask, out,
        H=H, W=W, total_rows=total_rows,
        stride_lm=linear_result.stride(0), stride_lh=linear_result.stride(1),
        stride_idx=idx_flat.stride(0),
        stride_sb=attn_scores.stride(0), stride_sh=attn_scores.stride(1),
        stride_sr=attn_scores.stride(2), stride_sc=attn_scores.stride(3),
        stride_mb=attn_mask.stride(0), stride_mr=attn_mask.stride(1),
        stride_mc=attn_mask.stride(2),
        stride_ob=out.stride(0), stride_oh=out.stride(1),
        stride_or=out.stride(2), stride_oc=out.stride(3),
        BLOCK_N=BLOCK_N,
    )

    return out


@torch.fx.wrap
def fused_linear_softmax(in_0, in_1, in_2, in_3, in_4):
    # Step 1: Compute linear = in_4 @ in_1.T
    x = in_4.view(-1, in_4.shape[-1])  # [225, 512]
    linear_result = triton_linear(x, in_1)  # [225, H]

    # Step 2: Fused bias computation + additions + softmax
    result = triton_fused_bias_add_softmax(linear_result, in_0, in_2, in_3)

    return result


def replacement_func():
    return fused_linear_softmax