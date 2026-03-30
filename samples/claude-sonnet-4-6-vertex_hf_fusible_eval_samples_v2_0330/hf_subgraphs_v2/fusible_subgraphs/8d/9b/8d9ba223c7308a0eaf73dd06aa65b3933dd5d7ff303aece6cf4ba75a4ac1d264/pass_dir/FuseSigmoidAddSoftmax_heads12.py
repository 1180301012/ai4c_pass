import torch
import triton
import triton.language as tl


def pattern(pos_bias, attn_scores, attn_mask):
    """
    Match: sigmoid(pos_bias) * 16 + attn_scores + 2*attn_mask -> softmax -> dropout
    For 12-head case: attn_scores shape [64, 12, 64, 64], attn_mask shape [64, 64, 64]
    """
    tmp_9 = torch.sigmoid(pos_bias)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = attn_scores + tmp_11
    tmp_13 = tmp_12.view(1, 64, 12, 64, 64)
    tmp_14 = attn_mask.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = attn_mask.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 12, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim = -1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return tmp_22


def replacement_args(pos_bias, attn_scores, attn_mask):
    return (pos_bias, attn_scores, attn_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['B', 'H'],
)
@triton.jit
def _fused_kernel_12head(
    pos_bias_ptr,
    attn_scores_ptr,
    attn_mask_ptr,
    out_ptr,
    B, H,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel: out[b,h,i,:] = softmax(attn_scores[b,h,i,:] + 16*sigmoid(pos_bias[h,i,:]) + 2*attn_mask[b,i,:])
    
    Grid: B * H * BLOCK_N programs (one per attention row)
    Each program handles BLOCK_N=64 elements (the key/col dimension)
    """
    pid = tl.program_id(0)

    # Decode batch, head, row indices from flat pid
    # pid = b * H * BLOCK_N + h * BLOCK_N + i
    b = pid // (H * BLOCK_N)
    h = (pid // BLOCK_N) % H
    i = pid % BLOCK_N

    # Column (softmax dimension) offsets
    j = tl.arange(0, BLOCK_N)

    # Load pos_bias[h, i, j]: shape [H, BLOCK_N, BLOCK_N]
    pb_offset = (h * BLOCK_N + i) * BLOCK_N + j
    pb = tl.load(pos_bias_ptr + pb_offset)
    pb_f32 = pb.to(tl.float32)
    # sigmoid and scale: 16 * sigmoid(pb)
    scaled_pb = tl.float32(16.0) / (tl.float32(1.0) + tl.exp(-pb_f32))

    # Load attn_scores[b, h, i, j]: shape [B, H, BLOCK_N, BLOCK_N]
    as_offset = ((b * H + h) * BLOCK_N + i) * BLOCK_N + j
    attn = tl.load(attn_scores_ptr + as_offset)
    attn_f32 = attn.to(tl.float32)

    # Load attn_mask[b, i, j]: shape [B, BLOCK_N, BLOCK_N]
    am_offset = (b * BLOCK_N + i) * BLOCK_N + j
    mask_val = tl.load(attn_mask_ptr + am_offset)
    mask_f32 = mask_val.to(tl.float32)

    # Fused: attn + 16*sigmoid(pos_bias) + 2*attn_mask
    x = attn_f32 + scaled_pb + tl.float32(2.0) * mask_f32

    # Softmax over j (dim=-1, size=BLOCK_N=64)
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    x_sm = x_exp / x_sum

    # Store result cast back to original dtype
    tl.store(out_ptr + as_offset, x_sm.to(attn.dtype))


@torch.fx.wrap
def fused_sigmoid_add_softmax_12head(pos_bias, attn_scores, attn_mask):
    """
    pos_bias:    [12, 64, 64]
    attn_scores: [64, 12, 64, 64]
    attn_mask:   [64, 64, 64]
    returns:     [64, 12, 64, 64]
    """
    B, H, SEQ_ROW, SEQ_COL = attn_scores.shape
    out = torch.empty_like(attn_scores)
    total_rows = B * H * SEQ_ROW  # 64 * 12 * 64 = 49152

    _fused_kernel_12head[(total_rows,)](
        pos_bias, attn_scores, attn_mask, out,
        B, H,
        BLOCK_N=64,
    )
    return out


def replacement_func():
    return fused_sigmoid_add_softmax_12head