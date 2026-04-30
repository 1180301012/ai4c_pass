import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel — weighted sum to compute x/y center-of-mass
#   Grid: (B * H,)  — one program per (batch, head) pair
#   Inputs:  tmp_3 [B,H,W,W] heatmap, in_0 [1,1,1,W] x-linspace, in_1 [1,1,H2,1] y-linspace
#   Output:  out_cm [B,H,1,2]
# ---------------------------------------------------------------------------

@triton.jit
def weighted_sum_kernel(
    heatmap_ptr,  # [B, H, N]
    x_ptr,        # [W]
    y_ptr,        # [H2]
    out_cm_ptr,   # [B*H, 2]
    B,
    H: tl.constexpr,
    W: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b   = pid // H
    h   = pid  % H
    offs = tl.arange(0, BLOCK)
    base = b * H * N + h * N

    vals   = tl.load(heatmap_ptr + base + offs).to(tl.float32)
    x_vals = tl.load(x_ptr + offs).to(tl.float32)
    y_vals = tl.load(y_ptr + offs).to(tl.float32)

    x_sum = tl.sum(vals * x_vals, axis=0)
    y_sum = tl.sum(vals * y_vals, axis=0)

    tl.store(out_cm_ptr + pid * 2 + 0, x_sum.to(out_cm_ptr.dtype.element_ty))
    tl.store(out_cm_ptr + pid * 2 + 1, y_sum.to(out_cm_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Wrapper — returns only tmp_10 (single output).
# tmp_3 is a pattern INPUT (not computed here), so we do NOT return it.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_weighted_sum(in_0, in_1, tmp_3):
    B  = tmp_3.shape[0]
    H  = 17
    W  = 64
    N  = 4096
    BLOCK = 4096

    out_cm = torch.empty(B, H, 1, 2, dtype=tmp_3.dtype, device=tmp_3.device)

    weighted_sum_kernel[(B * H,)](
        tmp_3, in_0, in_1,
        out_cm.view(B * H, 2),
        B=B, H=H, W=W, N=N, BLOCK=BLOCK,
    )

    return out_cm