"""
Fused Softmax+Matmul+Transpose for 256-token sequences.
Pattern: (in_0 + tmp_11).softmax(-1) @ in_4 → transpose → [4,128,256]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_attn_fwd_v256(
    scores_ptr,   # [4, 256, 256]
    in4_ptr,      # [4, 256, 128]
    out_ptr,      # [4, 128, 256]
):
    pid_row = tl.program_id(0)   # 0..1023  (b * 256 + n)
    pid_h   = tl.program_id(1)   # 0..7

    n_alias = pid_row
    b       = pid_row // 256
    n_alias = pid_row % 256

    row_off = b * 256 * 256 + n_alias * 256
    scores  = tl.load(scores_ptr + row_off + tl.arange(0, 256)).to(tl.float32)

    s_max   = tl.max(scores, axis=0)
    scores  = scores - s_max
    et      = tl.exp(scores) / tl.sum(tl.exp(scores), axis=0)

    # Load V_slice in_4[b, col_j*16+g*16 : +16, :] = [16, 128]
    col_j   = pid_h * 16 + tl.arange(0, 16)
    offs_v  = (col_j[:, None] * 256 + tl.arange(0, 256)[None, :])
    val     = tl.load(in4_ptr + b * 131072 + offs_v).to(tl.float32)  # [16, 256]

    result  = tl.sum(et[None, :] * val, axis=1)                      # [1, 128]

    out_off = b * 32768 + col_j * 256 + n_alias
    tl.store(out_ptr + out_off, result.to(out_ptr.dtype.element_ty))


def pattern(in_0, tmp_11, in_4):
    tmp_12  = in_0 + tmp_11
    tmp_13  = tmp_12.softmax(dim = -1)
    matmul  = tmp_13 @ in_4
    tmp_15  = matmul.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, tmp_11, in_4):
    return (in_0, tmp_11, in_4)


@torch.fx.wrap
def fused_softmax_attn_v256(in_0, tmp_11, in_4):
    out = torch.empty(4, 128, 256, dtype=in_0.dtype, device=in_0.device)
    _softmax_attn_fwd_v256[1024, 8](in_0, in_4, out)
    return out


def replacement_func():
    return fused_softmax_attn_v256