import torch
import triton
import triton.language as tl


# ── Pattern: matches scale-multiply only ──────────────────────────────────────
def pattern(in_0, in_1):
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.transpose(-2, -1)
    return (tmp_0, tmp_1)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Triton kernel: fused scale + transpose ────────────────────────────────────
@triton.jit
def fused_scale_transpose_kernel(
    in0_ptr,        # in_0 for transpose  [B, S, D] contiguous
    in1_ptr,        # in_1 for scale      [B, S, D] contiguous
    out_scale_ptr,  # scale output        [B, S, D]
    out_trans_ptr,  # trans output        [B, D, S]
    B,
    S,
    D,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Precompute row strides
    SD = S * D
    DS = D * S

    # ── Scale (in1/out_scale): flat index = b*SD + s*D + d ────────────────────
    b1 = offsets // SD
    rem1 = offsets % SD
    s1 = rem1 // D
    d1 = rem1 % D

    # ── Transpose (in0/out_trans): output flat index = b*DS + d*S + s ─────────
    b0 = offsets // DS
    rem0 = offsets % DS
    d0 = rem0 // S
    s0 = rem0 % S

    # Scale op: in_1[b,s,d] * scale → out_scale[b,s,d]
    in_1_vals = tl.load(in1_ptr + b1 * SD + s1 * D + d1, mask=mask, other=0.0)
    out_scale_vals = (in_1_vals.to(tl.float32) * 0.1767766952966369).to(in_1_vals.dtype)
    tl.store(out_scale_ptr + offsets, out_scale_vals, mask=mask)

    # Transpose op: in_0[b,s,d] → out_trans[b,d,s]
    in_0_vals = tl.load(in0_ptr + b1 * SD + s1 * D + d1, mask=mask, other=0.0)
    tl.store(out_trans_ptr + b0 * DS + d0 * S + s0, in_0_vals, mask=mask)


# ── Kernel wrapper ─────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_scale_transpose(in_0, in_1):
    B = in_0.shape[0]
    S = in_0.shape[1]
    D = in_0.shape[2]
    n_elements = B * S * D

    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out_scale = torch.empty_like(in_1)
    out_trans = torch.empty(B, D, S, device=in_0.device, dtype=in_0.dtype)

    fused_scale_transpose_kernel[(num_blocks,)](
        in_0, in_1,
        out_scale, out_trans,
        B, S, D,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_scale, out_trans


def replacement_func():
    return fused_scale_transpose