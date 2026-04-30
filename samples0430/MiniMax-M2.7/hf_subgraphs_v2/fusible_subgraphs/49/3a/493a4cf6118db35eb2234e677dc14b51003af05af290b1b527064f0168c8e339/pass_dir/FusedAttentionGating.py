import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_gating_kernel(
    in_0_ptr,       # gating params [16]
    in_1_ptr,       # patch_score [1, 16, 196, 196]
    in_2_ptr,       # pos_score [1, 16, 196, 196]
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_g: tl.constexpr,
):
    """
    Process softmax + gating computation.
    Grid: (H, W) where H=heads (16), W=rows (196)
    Each program processes one complete row of 196 elements.
    """
    pid_h = tl.program_id(0)
    pid_row = tl.program_id(1)
    
    # Compute base offset for this row
    head_offset = pid_h * W * W
    row_offset = pid_row * W
    base_offset = head_offset + row_offset
    
    # Load gating value for this head (scalar - broadcasts across the row)
    gating_val = tl.load(in_0_ptr + pid_h * stride_g).to(tl.float32)
    
    # Compute sigmoid once for this head
    sigmoid_pos = 1.0 / (1.0 + tl.exp(-gating_val))
    sigmoid_neg = 1.0 - sigmoid_pos
    
    # Load W elements for softmax computation
    offsets = tl.arange(0, 256)
    mask = offsets < W
    
    # Load all W elements for this row
    pos_scores = tl.load(in_2_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute softmax with numerical stability
    row_max = tl.max(pos_scores, axis=0)
    exp_scores = tl.exp(pos_scores - row_max)
    exp_sum = tl.sum(exp_scores, axis=0) + 1e-8
    softmax_vals = exp_scores / exp_sum
    
    # Load patch_score for this row
    patch_scores = tl.load(in_1_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute result: (1-sigmoid) * patch + sigmoid * softmax
    results = sigmoid_neg * patch_scores + sigmoid_pos * softmax_vals
    
    # Store results
    tl.store(out_ptr + base_offset + offsets, results, mask=mask)


@torch.fx.wrap
def fused_attention_gating(in_0, in_1, in_2):
    """
    Fused attention gating computation.
    Optimizes by:
    1. Computing sigmoid only once (avoiding redundant computation)
    2. Fusing softmax + gating + multiplication into single kernel
    3. Reducing memory bandwidth by keeping intermediate results in registers
    """
    H, W, _ = in_1.shape[1], in_1.shape[2], in_1.shape[3]  # [1, 16, 196, 196]
    
    out = torch.empty_like(in_1)
    
    # Grid: (heads, rows) = (16, 196)
    # Each program computes one complete row
    grid = (H, W)
    
    fused_attention_gating_kernel[grid](
        in_0, in_1, in_2, out,
        H=H, W=W, stride_g=1,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match the attention gating pattern:
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    """
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return tmp_8


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_attention_gating