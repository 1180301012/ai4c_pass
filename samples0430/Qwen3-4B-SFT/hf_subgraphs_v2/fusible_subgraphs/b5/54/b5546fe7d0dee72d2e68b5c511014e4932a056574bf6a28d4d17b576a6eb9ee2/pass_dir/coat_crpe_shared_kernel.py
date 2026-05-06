"""
Shared Triton kernel for the coat CRPE fused operation.
Fuses: cat -> reshape -> transpose -> mul -> pad -> scale+add -> transpose -> reshape
into a single memory-efficient kernel.

For each output element at (h+1, c_out) in the result:
  - in_4[h+1, c_out]  -> contributes in_4_val * scale
  - If h > 0:
      mul_val = concatted[c_out, h, :] * in_6[c_out, h, :]
  - result = mul_val + in_4_val * scale
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def coat_crpe_fused_kernel(
    in_2_ptr, in_3_ptr, conv_out_ptr, scale_ptr, in_4_ptr, in_6_ptr, output_ptr,
    C_in2, C_in3, C_conv, C_out, H, W, h_base,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Derive per-element indices from flat offset (batch=1)
    i = offsets
    h = i // (C_out * H * W)          # sequence position in [0, H+1)
    c_out = i % C_out                 # output channel

    # Scale: scale_ptr is a scalar (shape [1])
    scale_val = tl.load(scale_ptr)

    # Load in_4[i], then * scale  →  add to result
    in_4_val = tl.load(in_4_ptr + i, mask=mask, other=0.0)
    term_a = in_4_val * scale_val

    # Only positions h > 0 need mul from concatted/in_6
    use_mul   = (h > 0) & mask
    c_out_de  = c_out - h_base       # conv channel index

    # Safe channel indices: clamp to [0, C_conv-1] when h==0
    c_safe    = tl.where(c_out_de >= 0, c_out_de, 0)
    safe_in4  = i + H * C_out        # in_4[1,H+1,C_out] row-increment (unused clref)
    x6_offset = c_out_de * H * W     # in_6 channel stride in [1,8,H*W,W]

    # in_6: shape [1, 8, H*W, H_spatial=W]
    in_6_val = tl.load(in_6_ptr + i + x6_offset, mask=use_mul, other=0.0)

    # Padded concatted channel: row h-1 in raw [N, H, W] tensor
    h_m1       = tl.maximum(h - 1, 0)     # clamp to 0 for h=0 (avoids negative offset)
    x_raw_idx  = c_safe * H * W + h_m1 * W + tl.arange(0, BLOCK_SIZE)  # 1D

    # in_2 guard: only valid if h-1 >= 0 and c_out_de < C_in2
    guard_in2 = (use_mul & (c_out_de < C_in2))
    # in_3 guard: only valid if c_out_de < C_in3
    guard_in3 = (use_mul & (c_out_de < C_in3))
    use_conv  = use_mul & (c_out_de < C_conv)

    v2 = tl.load(in_2_ptr + x_raw_idx, mask=(use_mul & (c_out_de < C_in2)), other=0.0)
    v3 = tl.load(in_3_ptr + x_raw_idx, mask=(use_mul & (c_out_de < C_in3)), other=0.0)
    vc = tl.load(conv_out_ptr + x_raw_idx, mask=use_conv, other=0.0)
    conv_val = tl.where(c_out_de < C_in2, v2,
              tl.where(c_out_de < C_in3, v3, vc))

    result = in_6_val * conv_val + term_a
    tl.store(output_ptr + offsets, result, mask=mask)