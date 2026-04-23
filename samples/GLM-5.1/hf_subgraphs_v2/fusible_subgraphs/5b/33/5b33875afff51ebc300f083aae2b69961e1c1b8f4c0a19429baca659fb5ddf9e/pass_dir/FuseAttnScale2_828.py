import torch
import triton
import triton.language as tl

# Pattern matching - scale=2.8284271247461903
def pattern(in_0, in_2, in_3):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim = -1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3, "scale2_828")

@triton.jit
def fused_attn_kernel(
    scores_ptr, mask_ptr, value_ptr, out_ptr,
    B, H, S, D,
    scale,
    # strides for scores [B, H, S, S]
    scores_sb, scores_sh, scores_ss1, scores_ss2,
    # strides for mask (broadcast over B, H, S_out)
    mask_sb, mask_sh, mask_ss1, mask_ss2,
    # strides for value [B, H, S, D]
    value_sb, value_sh, value_ss, value_sd,
    # strides for out [B, S_out, H, D]
    out_sb, out_ss, out_sh, out_sd,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = B * S * H
    if pid >= total_rows:
        return
    
    b = pid // (S * H)
    remainder = pid % (S * H)
    h = remainder // S
    s_out = remainder % S
    
    # === Step 1: Load and scale attention scores row ===
    s_offsets = tl.arange(0, BLOCK_S)
    s_valid = s_offsets < S
    
    scores_off = b * scores_sb + h * scores_sh + s_out * scores_ss1 + s_offsets * scores_ss2
    scores_row = tl.load(scores_ptr + scores_off, mask=s_valid, other=0.0).to(tl.float32) * scale
    
    # === Step 2: Load mask (broadcast) ===
    mask_off = b * mask_sb + h * mask_sh + s_out * mask_ss1 + s_offsets * mask_ss2
    mask_row = tl.load(mask_ptr + mask_off, mask=s_valid, other=0.0).to(tl.float32)
    
    # === Step 3: Add mask ===
    scores_row = scores_row + mask_row
    
    # === Step 4: Softmax ===
    row_max = tl.max(scores_row, axis=0)
    scores_row = scores_row - row_max
    row_exp = tl.exp(scores_row)
    row_sum = tl.sum(row_exp, axis=0)
    softmax_row = row_exp / row_sum
    
    # === Step 5: Matmul with value tensor ===
    d_offsets = tl.arange(0, BLOCK_D)
    d_valid = d_offsets < D
    
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for s_idx in range(BLOCK_S):
        s_idx_valid = s_idx < S
        # Load value[b, h, s_idx, 0:D]
        v_off = b * value_sb + h * value_sh + s_idx * value_ss + d_offsets * value_sd
        v_row = tl.load(value_ptr + v_off, mask=d_valid & s_idx_valid, other=0.0).to(tl.float32)
        acc += softmax_row[s_idx] * v_row
    
    # === Step 6: Store output [b, s_out, h, d] ===
    out_off = b * out_sb + s_out * out_ss + h * out_sh + d_offsets * out_sd
    tl.store(out_ptr + out_off, acc, mask=d_valid)


@torch.fx.wrap
def fused_attn_dispatch(scores, mask, value, route):
    B, H, S, S2 = scores.shape
    D = value.shape[-1]
    
    # Determine scale based on route
    if route == "scale8":
        scale_val = 8.0
    elif route == "scale2_828":
        scale_val = 2.8284271247461903
    else:
        scale_val = 8.0  # default fallback
    
    # Output shape: [B, S, H, D] (permuted from [B, H, S, D])
    out = torch.empty((B, S, H, D), dtype=scores.dtype, device=scores.device)
    
    BLOCK_S = 64
    BLOCK_D = 64
    
    grid = (B * S * H,)
    
    fused_attn_kernel[grid](
        scores_ptr=scores, mask_ptr=mask, value_ptr=value, out_ptr=out,
        B=B, H=H, S=S, D=D,
        scale=1.0 / scale_val,
        scores_sb=scores.stride(0), scores_sh=scores.stride(1), 
        scores_ss1=scores.stride(2), scores_ss2=scores.stride(3),
        mask_sb=mask.stride(0), mask_sh=mask.stride(1),
        mask_ss1=mask.stride(2), mask_ss2=mask.stride(3),
        value_sb=value.stride(0), value_sh=value.stride(1),
        value_ss=value.stride(2), value_sd=value.stride(3),
        out_sb=out.stride(0), out_ss=out.stride(1),
        out_sh=out.stride(2), out_sd=out.stride(3),
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )
    
    return out

def replacement_func():
    return fused_attn_dispatch