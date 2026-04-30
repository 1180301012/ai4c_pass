import torch
import triton
import triton.language as tl


# ── pattern ───────────────────────────────────────────────────────────────────
# Match only the residual-add + in-place-relu_ that follow the linear.
# F.linear is left to cuBLAS; we fuse add+relu into one Triton kernel.
def pattern(linear, in_2):
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4


def replacement_args(linear, in_2):
    return (linear, in_2)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Fused residual-add + ReLU, no autotune (fixed grid = lower per-call overhead).
# For [M, N] = [1000, 128], N_ELEMENTS = 128000:
#   BLOCK=1024 → grid=(125,): best empirically on A30.
@triton.jit
def _fused_add_relu_kernel(
    a_ptr,     # residual  [M*N]
    b_ptr,     # linear    [M*N]
    out_ptr,   # output    [M*N]
    N_ELEMENTS,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEMENTS

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)

    c = a + b
    c = tl.maximum(c, 0.0)   # ReLU

    tl.store(out_ptr + offs, c, mask=mask)


# ── kernel wrapper ────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_add_relu(linear, in_2):
    """
    Computes relu(in_2 + linear) with a single Triton kernel.
    linear : [M, N]  – cuBLAS output, already on GPU
    in_2   : [M, N]  – residual tensor, already on GPU
    """
    out        = torch.empty_like(in_2)
    N_ELEMENTS = in_2.numel()   # 1000 * 128 = 128000

    # BLOCK=1024 → 125 blocks: empirically optimal for A30 with [1000,128].
    BLOCK = 1024
    grid  = (triton.cdiv(N_ELEMENTS, BLOCK),)

    _fused_add_relu_kernel[grid](
        in_2, linear, out,
        N_ELEMENTS,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=2,
    )

    return out


# ── replacement factory ───────────────────────────────────────────────────────
def replacement_func():
    return fused_add_relu