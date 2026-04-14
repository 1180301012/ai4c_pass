"""
Fused attention pass for 8-head, 32-dim attention (seq_len=1).

Mathematical insight: when seq_len=1,
  bmm(Q:[H,1,D], K^T:[H,D,1]) = [H,1,1]  (dot products)
  softmax([H,1,1], dim=-1) = [[1.0]] per head (single element)
  dropout(p=0.0) = identity
  bmm([[1.0]], V:[H,1,D]) = V (multiply by 1.0)
  view(1,H,1,D) → transpose(1,2) → reshape(1,1,H*D) = V.reshape(1,1,H*D)

So the entire subgraph reduces to: in_2.reshape(1, 1, 256)
"""

import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    bmm = torch.bmm(in_0, in_1)
    tmp_1 = torch.nn.functional.softmax(bmm, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    bmm_1 = torch.bmm(tmp_2, in_2)
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    # Only in_2 (value states) is needed; the math collapses to a reshape of in_2
    return (in_2,)


@triton.jit
def _copy_kernel_256(
    src_ptr,
    dst_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy 256 elements from src to dst in one block."""
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + offsets, x)


@torch.fx.wrap
def fused_attn_8h32d(in_2):
    """
    Optimized replacement: since seq_len=1, softmax collapses to 1.0,
    so the entire attention computation reduces to reshaping in_2
    from [8, 1, 32] to [1, 1, 256].
    """
    out = torch.empty((1, 1, 256), dtype=in_2.dtype, device=in_2.device)
    # 256 elements, single block launch
    _copy_kernel_256[(1,)](in_2, out, BLOCK_SIZE=256)
    return out


def replacement_func():
    return fused_attn_8h32d