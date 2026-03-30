import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4, tmp_2)


# ---------------------------------------------------------------------------
# Triton kernel
#
# Grid = (2,):
#   pid 0  →  normalise in_1  → write tmp_2
#   pid 1  →  normalise in_2  → write tmp_4 & tmp_6
# ---------------------------------------------------------------------------
@triton.jit
def fused_l2norm_exp_kernel(
    in0_ptr,        # scalar tensor  (the logit scale)
    in1_ptr,        # [N] flat view of in_1
    in2_ptr,        # [N] flat view of in_2
    out_tmp2_ptr,   # [N] flat output for tmp_2
    out_tmp4_ptr,   # [N] flat output for tmp_4
    out_tmp6_ptr,   # [N] flat output for tmp_6
    N,              # number of elements (runtime, same for both vectors)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    if pid == 0:
        # ── Normalise in_1 ──────────────────────────────────────────────────
        x_raw = tl.load(in1_ptr + offs, mask=mask, other=0.0)
        x     = x_raw.to(tl.float32)
        norm  = tl.sqrt(tl.sum(x * x, axis=0))
        xn    = x / norm
        tl.store(out_tmp2_ptr + offs, xn.to(x_raw.dtype), mask=mask)

    else:
        # ── Normalise in_2, then scale by exp(in_0) ─────────────────────────
        x_raw    = tl.load(in2_ptr + offs, mask=mask, other=0.0)
        x        = x_raw.to(tl.float32)
        norm     = tl.sqrt(tl.sum(x * x, axis=0))
        xn       = x / norm

        in0_val  = tl.load(in0_ptr).to(tl.float32)
        scale    = tl.exp(in0_val)

        tl.store(out_tmp4_ptr + offs, xn.to(x_raw.dtype), mask=mask)
        tl.store(out_tmp6_ptr + offs, (xn * scale).to(x_raw.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper  (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_normalize_and_scale(in_0, in_1, in_2):
    """
    in_0  : scalar tensor (logit scale)
    in_1  : [1, 512]
    in_2  : [1, 1, 512]
    Returns (tmp_6, tmp_4, tmp_2) matching the original graph outputs.
    """
    N     = in_1.shape[-1]                      # 512
    BLOCK = triton.next_power_of_2(N)           # 512

    in1_flat = in_1.contiguous().reshape(-1)    # [512]
    in2_flat = in_2.contiguous().reshape(-1)    # [512]

    tmp_2_flat = torch.empty_like(in1_flat)
    tmp_4_flat = torch.empty_like(in2_flat)
    tmp_6_flat = torch.empty_like(in2_flat)

    fused_l2norm_exp_kernel[(2,)](
        in_0,
        in1_flat,
        in2_flat,
        tmp_2_flat,
        tmp_4_flat,
        tmp_6_flat,
        N,
        BLOCK=BLOCK,
    )

    tmp_2 = tmp_2_flat.reshape(in_1.shape)
    tmp_4 = tmp_4_flat.reshape(in_2.shape)
    tmp_6 = tmp_6_flat.reshape(in_2.shape)

    return (tmp_6, tmp_4, tmp_2)


# ---------------------------------------------------------------------------
# Required interface functions
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_normalize_and_scale