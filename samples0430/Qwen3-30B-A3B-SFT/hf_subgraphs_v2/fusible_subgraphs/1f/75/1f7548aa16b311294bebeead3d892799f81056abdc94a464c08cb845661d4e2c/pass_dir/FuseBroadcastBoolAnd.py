import torch
import operator
import triton
import triton.language as tl


@triton.jit
def _bool_broadcast_and_kernel(
    b1_ptr,   # [Nq, Nk] bool – causal mask
    b2_ptr,   # [B, Nk]  bool – attention mask
    out_ptr,  # [B, 1, Nq, Nk] bool
    B, Nq, Nk,
    BLOCK_N: tl.constexpr,
):
    """
    Fused broadcast-AND:
      out[b, 0, i, j] = b1[i, j] & b2[b, j]
    Grid: (B * Nq,)
    """
    pid = tl.program_id(0)
    b = pid // Nq
    i = pid % Nq
    j_offs = tl.arange(0, BLOCK_N)
    mask = j_offs < Nk
    row1 = tl.load(b1_ptr + i * Nk + j_offs, mask=mask, other=False)
    row2 = tl.load(b2_ptr + b * Nk + j_offs, mask=mask, other=False)
    result = row1 & row2
    tl.store(out_ptr + pid * Nk + j_offs, result, mask=mask)


@torch.fx.wrap
def fused_bool_broadcast_and(b1, b2):
    """
    b1: [Nq, Nk] bool
    b2: [B, Nk]  bool
    out: [B, 1, Nq, Nk] bool  == b1[:, None, :] & b2[None, :, None, :]
    """
    Nq, Nk = b1.shape
    B = b2.shape[0]
    out = torch.empty((B, 1, Nq, Nk), dtype=torch.bool, device=b1.device)
    grid = (B * Nq,)
    _bool_broadcast_and_kernel[grid](b1, b2, out, B, Nq, Nk, BLOCK_N=1024)
    return out


def pattern(causal_le, mask_2d):
    tmp_10 = causal_le[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = mask_2d[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    return tmp_13


def replacement_args(causal_le, mask_2d):
    return (causal_le, mask_2d)


def replacement_func():
    return fused_bool_broadcast_and