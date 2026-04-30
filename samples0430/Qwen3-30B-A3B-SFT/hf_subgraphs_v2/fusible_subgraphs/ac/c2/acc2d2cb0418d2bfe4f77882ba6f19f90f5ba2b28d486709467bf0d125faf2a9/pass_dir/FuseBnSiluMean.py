import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: batch_norm (inference) → silu → mean over spatial dims
# This pattern is identical across ALL target graphs.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(tmp_8, tmp_0, tmp_1, tmp_3, tmp_2):
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.silu(tmp_9, inplace=True)
    tmp_11 = tmp_10.mean((2, 3), keepdim=True)
    return (tmp_10, tmp_11)


def replacement_args(tmp_8, tmp_0, tmp_1, tmp_3, tmp_2):
    # (input, running_mean, running_var, weight, bias)
    return (tmp_8, tmp_0, tmp_1, tmp_3, tmp_2)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused BN-inference + SiLU + spatial mean (H, W)
#
# Grid: (N * C,)  — one program per (batch, channel) pair.
# Each program iterates over H*W spatial elements in BLOCK_HW tiles,
# accumulates the sum for the mean, and stores one output scalar.
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}),
        triton.Config({'BLOCK_HW': 128}),
        triton.Config({'BLOCK_HW': 256}),
        triton.Config({'BLOCK_HW': 512}),
        triton.Config({'BLOCK_HW': 1024}),
    ],
    key=['HW'],
)
@triton.jit
def _fused_bn_silu_mean_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    silu_out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # ── load per-channel BN parameters (always fp32 for stability) ──
    running_mean = tl.load(mean_ptr    + c).to(tl.float32)
    running_var  = tl.load(var_ptr     + c).to(tl.float32)
    w            = tl.load(weight_ptr  + c).to(tl.float32)
    b            = tl.load(bias_ptr    + c).to(tl.float32)

    eps = 1e-5
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    scale   = w * inv_std
    shift   = b - running_mean * scale   # fuse mean-subtraction into shift

    base = (n * C + c) * HW
    acc  = tl.zeros([1], dtype=tl.float32)

    # ── iterate over spatial tiles ──
    for start in range(0, HW, BLOCK_HW):
        offsets = start + tl.arange(0, BLOCK_HW)
        mask    = offsets < HW

        # load input; masked positions → 0.0 (contributes 0 to sum & SiLU)
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

        # BN inference
        x = x * scale + shift

        # SiLU:  x * sigmoid(x)
        silu_x = x * tl.sigmoid(x)

        # store SiLU output (Triton auto-casts fp32→fp16/bf16 per pointer dtype)
        tl.store(silu_out_ptr + base + offsets, silu_x, mask=mask)

        # accumulate only valid (unmasked) elements
        acc += tl.sum(tl.where(mask, silu_x, 0.0), axis=0)

    # compute mean and store
    mean_val = acc / HW
    tl.store(mean_ptr + n * C + c, mean_val)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper  (must be decorated with @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_bn_silu_mean(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    HW = H * W

    silu_out = torch.empty_like(x)
    # mean output shape: (N, C, 1, 1) — contiguous, same dtype as x
    mean_out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)

    _fused_bn_silu_mean_kernel[(N * C,)](
        x,
        mean_out,
        running_mean,
        running_var,
        weight,
        bias,
        silu_out,
        C,
        HW,
    )

    return silu_out, mean_out


def replacement_func():
    return fused_bn_silu_mean