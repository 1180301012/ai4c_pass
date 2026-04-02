"""Fuses: 0.1767766952966369 * in4 + padded -> transpose(1,2)  [K=32]"""
import torch
import triton
import triton.language as tl

_SCALE = 0.1767766952966369

def pattern(in4, padded):
    tmp8  = 0.1767766952966369 * in4
    tmp9  = tmp8 + padded
    tmp10 = tmp9.transpose(1, 2)
    return tmp10

def replacement_args(in4, padded):
    return (in4, padded)

@triton.jit
def _fma_transpose_K32(
    in4_ptr, pad_ptr, out_ptr,
    H, HW1, K,
    s_in4_h, s_in4_i, s_in4_k,
    s_pad_h, s_pad_i, s_pad_k,
    s_out_i, s_out_h, s_out_k,
    SCALE: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H;  i = pid // H
    k = tl.arange(0, BLOCK_K);  mask = k < K
    in4_val = tl.load(in4_ptr + h*s_in4_h + i*s_in4_i + k*s_in4_k, mask=mask, other=0.0)
    pad_val = tl.load(pad_ptr + h*s_pad_h + i*s_pad_i + k*s_pad_k, mask=mask, other=0.0)
    tl.store(out_ptr + i*s_out_i + h*s_out_h + k*s_out_k, SCALE*in4_val + pad_val, mask=mask)

@torch.fx.wrap
def fused_fma_transpose_K32(in4, padded):
    B, H, HW1, K = in4.shape
    out = torch.empty(B, HW1, H, K, dtype=in4.dtype, device=in4.device)
    _fma_transpose_K32[(H*HW1,)](
        in4, padded, out, H, HW1, K,
        in4.stride(1), in4.stride(2), in4.stride(3),
        padded.stride(1), padded.stride(2), padded.stride(3),
        out.stride(1), out.stride(2), out.stride(3),
        SCALE=_SCALE, BLOCK_K=32,
    )
    return out

def replacement_func():
    return fused_fma_transpose_K32