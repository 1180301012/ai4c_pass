import torch
import triton
import triton.language as tl


# ─── Pattern ────────────────────────────────────────────────────────────────

def pattern(linear_out, in_2):
    """
    linear_out : [1, 12, 199, 8]  — output of F.linear(in_3, weight, bias)
    in_2       : [1, 12, 1, 1]    — rel-pos const (scale)
    """
    tmp_4 = linear_out.view(1, 12, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 12, -1, 1)
    return tmp_14


def replacement_args(linear_out, in_2):
    return (linear_out, in_2)


# ─── Triton kernel ───────────────────────────────────────────────────────────
# Fixed-shape version for N=12, S=199, D=8 (the two groups dim = 2).
# pid encodes a single (head n, seq s) pair; load 2×2=4 elements per program.

@triton.jit
def _fused_rel_pos_kernel_12(
    linear_ptr,   # [12, 199, 4] fp16/bf16
    in_2_ptr,     # [12]        fp16/bf16 (per-head scale)
    out_ptr,      # [12 * 199]  fp16/bf16
):
    pid = tl.program_id(0)
    n   = pid // 199
    s   = pid - n * 199

    offsets = s * 2 + tl.arange(0, 2)   # load 2 of 4 elements

    # Load 2 of 4 linear outputs (in_2 has stride 1 for [N,1,1])
    y   = tl.load(linear_ptr + n * 199 * 4 + s * 2 + offsets).to(tl.float32)
    sc  = tl.load(in_2_ptr    +              n   ).to(tl.float32)

    # sigmoid( sum of all 4 )  —  both chunk halves are s0/2
    s0  = tl.sigmoid(tl.sum(y,  axis=0))
    c0  = s0 * 0.5
    c1  = s0 * 0.5

    # (s0/2)*scale - 1  +  (s0/2)*scale*2
    out = c0 * sc - 1.0 + c1 * sc * 2.0

    tl.store(out_ptr + n * 199 + s, out)


# ─── Wrapper ────────────────────────────────────────────────────────────────

@torch.fx.wrap
def triton_rel_pos_12(linear_out, in_2):
    """
    linear_out: [1, 12, 199, 8],  in_2: [1, 12, 1, 1]
    output:     [1, 12, 199, 1]
    """
    N = 12
    S = 199
    D = 8
    D2 = 4    # D // 2

    out = torch.empty((1, N, S, 1), dtype=linear_out.dtype, device=linear_out.device)

    _fused_rel_pos_kernel_12[(N * S,)](
        linear_out,   # treats [1, 12, 199, 8] as contiguous [12*199*8]
        in_2,         # treats [1, 12, 1, 1]   as contiguous [12]
        out,
        num_warps=2,
    )

    return out


# ─── Replacement hook ────────────────────────────────────────────────────────

def replacement_func():
    return triton_rel_pos_12