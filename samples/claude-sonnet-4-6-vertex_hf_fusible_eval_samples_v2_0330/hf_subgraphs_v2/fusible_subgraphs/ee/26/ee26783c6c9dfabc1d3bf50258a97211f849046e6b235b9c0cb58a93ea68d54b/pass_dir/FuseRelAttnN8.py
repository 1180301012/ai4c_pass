"""
Pass: FuseRelAttnN8
Fuses the relative-position attention bias computation (N=8 spatial, 8 heads)
with softmax for the botnet26t 8x8 subgraph.

Pattern starts from tmp_7 (reshape of the slice result) to avoid pad-matching issues.
Inputs:
  in_0     : attention bias [4, 64, 64]
  in_2     : relative logits [4, 8, 8, 8, 8]
  in_4     : value matrix [4, 64, 128]
  tmp_6_in : slice result [32, 8, 8] (non-contiguous view, so we call .contiguous())

Key index mapping for attn[B, I, J]:
  a1=I//8, a2=I%8, a3=J//8, a4=J%8
  s_val = tmp_6_in_c[B*8+a2, a1, a3]   (contiguous [32,8,8], strides [64,8,1])
  in2_val = in_2[B, a1, a2, a3, a4]  (= in2_base + J for fixed B,I)
  attn[B,I,J] = in_0[B,I,J] + in2_val + s_val
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused rel-pos bias + softmax   (N=8, seq=64)
# Reads tmp_6_in (contiguous [32,8,8], strides [64,8,1]),
# in_2 (contiguous [4,8,8,8,8], strides [4096,512,64,8,1]),
# in_0 (contiguous [4,64,64], strides [4096,64,1]).
# Writes float32 softmax to out_ptr.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_rel_attn_softmax_n8(
    s6_ptr,   # tmp_6_in contiguous [32, 8, 8]
    in2_ptr,  # in_2 contiguous [4, 8, 8, 8, 8]
    in0_ptr,  # in_0 contiguous [4, 64, 64]
    out_ptr,  # float32 output [4, 64, 64]
):
    # ---------- grid: one program per (B, I) row ----------
    pid    = tl.program_id(0)
    B_idx  = pid >> 6          # pid // 64
    I      = pid & 63          # pid % 64

    a1 = I >> 3                # I // 8
    a2 = I & 7                 # I % 8

    # Column indices
    J  = tl.arange(0, 64)      # [0 .. 63]
    a3 = J >> 3                # J // 8  – only 8 distinct values

    # ---- s_val from tmp_6_in: [32,8,8] strides [64,8,1]
    # s_val[J] = tmp_6_in[B*8+a2, a1, a3]
    s6_off = (B_idx * 8 + a2) * 64 + a1 * 8 + a3
    s_val  = tl.load(s6_ptr + s6_off).to(tl.float32)

    # ---- in_2: [4,8,8,8,8] strides [4096,512,64,8,1]
    # in_2[B, a1, a2, a3, a4] = base + J  (a3*8+a4 = J)
    in2_base = B_idx * 4096 + a1 * 512 + a2 * 64
    in2_val  = tl.load(in2_ptr + in2_base + J).to(tl.float32)

    # ---- in_0: [4,64,64] strides [4096,64,1]
    in0_val  = tl.load(in0_ptr + B_idx * 4096 + I * 64 + J).to(tl.float32)

    # ---- attention logit ----
    attn = in0_val + in2_val + s_val

    # ---- numerically-stable softmax ----
    m    = tl.max(attn, axis=0)
    attn = attn - m
    e    = tl.exp(attn)
    sm   = e / tl.sum(e, axis=0)

    # ---- store float32 ----
    tl.store(out_ptr + B_idx * 4096 + I * 64 + J, sm)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_rel_attn_n8(in_0, in_2, in_4, tmp_6_in):
    """
    Fused: reshape+expand+permute+add(in_2)+reshape+add(in_0)+softmax+matmul+transpose
    for the N=8 botnet subgraph.

    tmp_6_in : [32, 8, 8] slice result (may be non-contiguous)
    in_0     : [4, 64, 64]
    in_2     : [4, 8, 8, 8, 8]
    in_4     : [4, 64, 128]
    returns  : [4, 128, 64]
    """
    B, seq = 4, 64

    # Ensure contiguous inputs
    s6c  = tmp_6_in.contiguous()   # [32, 8, 8]
    in2c = in_2.contiguous()        # [4, 8, 8, 8, 8]
    in0c = in_0.contiguous()        # [4, 64, 64]

    # Softmax output in float32
    softmax_f32 = torch.empty((B, seq, seq), dtype=torch.float32, device=in_0.device)

    _fused_rel_attn_softmax_n8[(B * seq,)](
        s6c, in2c, in0c, softmax_f32,
        num_warps=2,
    )

    # Convert back to input dtype, then final matmul + transpose
    softmax_out = softmax_f32.to(in_0.dtype)
    out = softmax_out @ in_4          # [4, 64, 128]
    return out.transpose(-1, -2)      # [4, 128, 64]


# ---------------------------------------------------------------------------
# Pattern / replacement hooks
# ---------------------------------------------------------------------------

def pattern(in_0, in_2, in_4, tmp_6_in):
    tmp_7    = tmp_6_in.reshape(4, 8, 1, 8, 8)
    tmp_8    = tmp_7.expand(-1, -1, 8, -1, -1)
    tmp_9    = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10   = tmp_9 + in_2
    tmp_11   = tmp_10.reshape(4, 64, 64)
    tmp_12   = in_0 + tmp_11
    tmp_13   = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15   = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, in_2, in_4, tmp_6_in):
    return (in_0, in_2, in_4, tmp_6_in)


def replacement_func():
    return fused_rel_attn_n8