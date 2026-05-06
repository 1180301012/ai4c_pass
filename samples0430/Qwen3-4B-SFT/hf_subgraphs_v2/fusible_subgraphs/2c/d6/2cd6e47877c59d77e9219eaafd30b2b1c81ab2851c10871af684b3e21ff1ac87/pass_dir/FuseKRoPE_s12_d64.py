"""
Pass: FuseKRoPE_s12_d64
Matches the Key Rope subgraph for eva02_base_patch16_clip (H=12, S=196).

Pattern:
  tmp_11 = in_4[ ..., :, 0, ... ]
  tmp_12 = in_4[ ..., :, 1:, ... ]
  split  = in_0.tensor_split(2, -1)
  tmp_14 = split[0]   # xh
  tmp_15 = split[1]   # xs
  tmp_16 = tmp_12 * tmp_15
  tmp_17 = tmp_12[..., 1::2]
  tmp_18 = -tmp_17
  tmp_19 = tmp_12[..., ::2]
  tmp_20 = stack([tmp_18, tmp_19], -1)
  tmp_21 = tmp_20.reshape((1, 12, 196, 64))
  tmp_22 = tmp_21 * tmp_14
  tmp_23 = tmp_16 + tmp_22
  tmp_24 = cat([tmp_11, tmp_23], dim=2)
  tmp_25 = tmp_24.type_as(in_6)
  return tmp_25   # shape [1, 12, 197, 64]
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel_k(
    x_ptr,       # [1, H, S+1, D]  (in_4) – keep only the last S rows
    pos_ptr,     # [N=2S, D]       (in_0)  – full pos_embed, we use [S, :] for last tokens
    out_ptr,     # [1, H, S, D]
    H,
    S,
    BLOCK_D: tl.constexpr,
):
    h       = tl.program_id(0)
    s       = tl.program_id(1)
    d_start = tl.program_id(2) * BLOCK_D

    d_offs   = tl.arange(0, BLOCK_D)
    d_global = d_start + d_offs

    # Load pos_embed rows for positions [s, s+1//2, ..., s+1-2] → the D cols at (last_token_index_s) = s+0*S_half where S_half = S for [N,S,D] shaped embedding
    # in_0 has shape [N, 128]; for eva02 N=S*2, N//2 = S, cols=[64:128]
    pos_row = tl.load(pos_ptr + tl.arange(0, 2) * S + d_global)

    # Load x patch row (in_4[:, :, 1:, :])
    x_patch = tl.load(x_ptr + h * (S + 1) * BLOCK_D + (s + 1) * BLOCK_D + d_global)

    # path B RoPE:
    #   out_even  = x_patch_even * cos_even  - x_patch_odd * sin_even
    #   out_odd   = x_patch_even * sin_even  + x_patch_odd * cos_even
    # i.e. out = rotate_half(x_patches, cos=S[:, :, :64], sin=S[:, :, 64:128])

    x_even = tl.load(x_ptr + h * (S + 1) * BLOCK_D + (s + 1) * BLOCK_D + d_global,
                     mask=d_global % 2 == 0, other=0.0)
    x_odd  = tl.load(x_ptr + h * (S + 1) * BLOCK_D + (s + 1) * BLOCK_D + d_global,
                     mask=d_global % 2 == 1, other=0.0)

    pos_even = tl.load(pos_ptr + tl.arange(0, 2) * S + d_global,
                       mask=d_global % 2 == 0, other=0.0)
    pos_odd  = tl.load(pos_ptr + tl.arange(0, 2) * S + d_global,
                       mask=d_global % 2 == 1, other=0.0)

    rope_e   = x_even * pos_even - x_odd * pos_odd
    rope_o   = x_odd  * pos_odd  + x_even * pos_even

    output_base = h * S * BLOCK_D + s * BLOCK_D
    tl.store(out_ptr + output_base + tl.arange(0, BLOCK_D // 2),
             rope_e,  mask=tl.arange(0, BLOCK_D // 2) < BLOCK_D // 2)
    tl.store(out_ptr + output_base + BLOCK_D // 2 + tl.arange(0, BLOCK_D // 2),
             rope_o,  mask=tl.arange(0, BLOCK_D // 2) < BLOCK_D // 2)


def pattern(in_0, in_4, in_6):
    tmp_11 = in_4[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
    tmp_12 = in_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    split  = in_0.tensor_split(2, -1)
    tmp_14 = split[0]
    tmp_15 = split[1]
    tmp_16 = tmp_12 * tmp_15
    tmp_17 = tmp_12[(Ellipsis, slice(1, None, 2))]
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[(Ellipsis, slice(None, None, 2))]
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape((1, 12, 196, 64))
    tmp_22 = tmp_21 * tmp_14
    tmp_23 = tmp_16 + tmp_22
    tmp_24 = torch.cat([tmp_11, tmp_23], dim=2)
    tmp_25 = tmp_24.type_as(in_6)
    return tmp_25


def replacement_args(in_0, in_4, in_6):
    return (in_0, in_4, in_6)


@torch.fx.wrap
def _kr_t12s196(pos_embed, k_full, ref):
    """
    pos_embed : [N, 128]  = in_0
    k_full    : [1, H, S+1, D] = in_4
    ref       : [1, H, S+1, D] (dtype)
    returns   : [1, H, S, D]
    """
    H = k_full.shape[1]
    S = k_full.shape[2] - 1
    D = k_full.shape[3]
    rope_out = torch.empty((1, H, S, D), dtype=ref.dtype, device=ref.device)
    k_full = k_full.view(H, S + 1, D)
    rope_out = rope_out.view(H, S, D)
    grid = (H, S, triton.cdiv(D, 64))
    _rope_kernel_k[grid](k_full, pos_embed, rope_out, H, S, BLOCK_D=64)
    return rope_out


def replacement_func():
    return _kr_t12s196