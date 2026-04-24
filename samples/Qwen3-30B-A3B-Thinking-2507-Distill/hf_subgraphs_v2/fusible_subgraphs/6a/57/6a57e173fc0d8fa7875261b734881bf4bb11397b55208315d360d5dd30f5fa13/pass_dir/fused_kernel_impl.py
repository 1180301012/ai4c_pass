"""
Shared fused depthwise-conv2d + in-place-add + permute + contiguous kernel.

Matches the pattern:
  conv_out = conv2d(in_2, weight, None, (1,1), (32,0), (1,1), groups)
  in_1 += conv_out           # iadd
  tmp   = in_1.permute(0,2,1,3)
  out   = tmp.contiguous()
  result = out.view(B, H, C*D)

Fused implementation: for each output element [b,h,c,d] compute the depthwise
convolution, add in_1[b,c,h,d], and write directly to the permuted layout
[b,h,c,d] — no intermediate conv-output buffer needed.

Weight shape: [C, 1, K, 1] where K=65 is fixed for all observed graphs.
Input shapes: [B, C, H, D] with D=8 or D=64.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 64}, num_warps=8),
    ],
    key=['B', 'C', 'H', 'D'],
)
@triton.jit
def fused_dw_conv_add_permute_kernel(
    in1_ptr,      # [B, C, H, D]
    in2_ptr,      # [B, C, H, D]
    wt_ptr,       # [C, 1, K, 1]
    out_ptr,      # [B, H, C*D]   -- final permuted + flattened output
    B, C, H, D,
    K:      tl.constexpr,   # kernel height = 65 (fixed)
    BLOCK_H: tl.constexpr,
):
    # ---- decode program id -----------------------------------------------
    pid = tl.program_id(0)   # indexes B * C
    n_h_blocks = tl.cdiv(H, BLOCK_H)
    pid2 = tl.program_id(1)  # indexes ceil(H/BLOCK_H)

    b_idx = pid // C
    c_idx = pid % C
    h_start = pid2 * BLOCK_H

    # ---- index vectors ---------------------------------------------------
    h_offsets = h_start + tl.arange(0, BLOCK_H)   # [BLOCK_H]
    d_offsets = tl.arange(0, D)                    # [D]   (D constexpr? no, but tl.arange works)

    h_mask = h_offsets < H                         # [BLOCK_H]
    d_mask = d_offsets < D                         # [D]

    h2d_mask = h_mask[:, None] & d_mask[None, :]  # [BLOCK_H, D]

    # ---- main convolution loop (K=65, unrolled by Triton as constexpr) ---
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    for k in range(K):                             # K is constexpr → loop is unrolled
        # Pad: zero outside [0, H-1]
        h_in = h_offsets + (K // 2) - k            # [BLOCK_H]
        valid_h = h_in >= 0, h_in < H
        h_in_safe = tl.where(valid_h[0] & valid_h[1], h_in, 0)

        in2_idx = (b_idx * C * H * D
                   + c_idx * H * D
                   + h_in_safe[:, None] * D
                   + d_offsets[None, :])            # [BLOCK_H, D]
        in2_val = tl.load(in2_ptr + in2_idx, mask=h2d_mask & valid_h[0][:, None] & valid_h[1][:, None], other=0.0)

        # weight[c, 0, k, 0]  (shape [C,1,K,1] → stride = [K, K, 1, 1])
        wt_val = tl.load(wt_ptr + c_idx * K + k)   # scalar

        acc += in2_val.to(tl.float32) * wt_val.to(tl.float32)

    # ---- fused add-in1 & permuted write -----------------------------------
    base = b_idx * C * H * D + c_idx * H * D      # [B,C,H,D] base for this (b,c)
    in1_idx = base + h_offsets[:, None] * D + d_offsets[None, :]   # [BLOCK_H, D]
    in1_val = tl.load(in1_ptr + in1_idx, mask=h2d_mask, other=0.0)

    result = acc + in1_val.to(tl.float32)          # [BLOCK_H, D]

    # Output layout [B, H, C, D] → flattened last two dims → [B, H, C*D]
    out_idx = (b_idx * H * C * D
               + h_offsets[:, None] * C * D
               + c_idx * D
               + d_offsets[None, :])               # [BLOCK_H, D]
    tl.store(out_ptr + out_idx, result.to(in1_val.dtype), mask=h2d_mask)


@torch.fx.wrap
def fused_conv_add_permute_view(in_0, in_1, in_2):
    """
    in_0 : weight  [C, 1, K, 1]
    in_1 : context [B, C, H, D]
    in_2 : value   [B, C, H, D]
    Returns: [B, H, C*D]
    """
    B, C, H, D = in_1.shape
    K = 65
    out = torch.empty(B, H, C * D, dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (B * C, triton.cdiv(H, meta['BLOCK_H']))

    fused_dw_conv_add_permute_kernel[grid](
        in_1, in_2, in_0, out,
        B, C, H, D,
        K=K,
    )
    return out