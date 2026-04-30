"""
Shared Triton kernels and dispatch wrapper for the fused BN+Add passes.
"""
import torch
import triton
import triton.language as tl
import threading

# Thread-local cache to avoid running the full kernel twice.
_dispatch_cache = threading.local()


# ── Kernel 1: fused batch_norm (inference) + residual add ─────────────────────
# Computes: output = BN(input) + residual   (no relu — relu is applied by model)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _bn_add_kernel(
    in4_ptr, in0_ptr, in1_ptr, in3_ptr, in2_ptr, in5_ptr, out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """BN (inference) + residual add → write to out_ptr (native dtype)."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    rm = tl.load(in0_ptr + pid_c).to(tl.float32)
    rv = tl.load(in1_ptr + pid_c).to(tl.float32)
    rg = tl.load(in3_ptr + pid_c).to(tl.float32)
    rb = tl.load(in2_ptr + pid_c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale   = rg * inv_std
    shift   = rb - rm * scale

    base = pid_b * C * HW + pid_c * HW

    for i in range(tl.cdiv(HW, BLOCK_HW)):
        hw_off = i * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask   = hw_off < HW
        x = tl.load(in4_ptr + base + hw_off, mask=mask, other=0.0)
        r = tl.load(in5_ptr + base + hw_off, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        rf = r.to(tl.float32)
        z  = scale * xf + shift + rf     # BN + add (no relu)
        tl.store(out_ptr + base + hw_off, z.to(x.dtype), mask=mask)


# ── Kernel 2: spatial mean over (H, W) ───────────────────────────────────────

@triton.jit
def _mean_kernel(
    z_ptr, mean_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    base = pid_b * C * HW + pid_c * HW
    total = 0.0

    for i in range(tl.cdiv(HW, BLOCK_HW)):
        hw_off = i * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask   = hw_off < HW
        z = tl.load(z_ptr + base + hw_off, mask=mask, other=0.0)
        total = total + tl.sum(z.to(tl.float32))

    tl.store(mean_ptr + pid_b * C + pid_c, (total / HW).to(tl.float32))


# ── Python helpers ────────────────────────────────────────────────────────────

@torch.fx.wrap
def run_bn_add(in_0, in_1, in_2, in_3, in_4, in_5):
    """Fused BN (inference) + residual add. Returns relu_out only."""
    B, C, H, W = in_4.shape
    HW  = H * W
    eps = 1e-5
    relu_out = torch.empty_like(in_4)

    _bn_add_kernel[(B, C)](
        in_4, in_0, in_1, in_3, in_2, in_5, relu_out,
        C, HW, eps,
    )
    return relu_out


@torch.fx.wrap
def run_compute_mean(x):
    """Compute spatial mean of x. Used by the mean-only route."""
    B, C, H, W = x.shape
    HW  = H * W
    mean_out = torch.empty((B, C, 1, 1), dtype=torch.float32, device=x.device)
    MBLOCK = 1024
    _mean_kernel[(B, C)](x, mean_out, C, HW, BLOCK_HW=MBLOCK)
    return x, mean_out


# ── Shared dispatch wrapper ────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_wrapper(*args):
    """
    Route-based dispatch.  Last arg is a route string.
    """
    last_arg = args[-1]
    if last_arg == "bn_add":
        in_0, in_1, in_2, in_3, in_4, in_5 = args[:6]
        B, C, H, W = in_4.shape
        relu_out = torch.empty_like(in_4)
        _bn_add_kernel[(B, C)](
            in_4, in_0, in_1, in_3, in_2, in_5, relu_out,
            C, H * W, 1e-5,
        )
        return relu_out
    elif last_arg == "mean_only":
        x = args[0]
        B, C, H, W = x.shape
        mean_out = torch.empty((B, C, 1, 1), dtype=torch.float32, device=x.device)
        _mean_kernel[(B, C)](x, mean_out, C, H * W, BLOCK_HW=1024)
        return x, mean_out
    # Fallback: assume args are BN+add inputs
    in_0, in_1, in_2, in_3, in_4, in_5 = args[:6]
    B, C, H, W = in_4.shape
    relu_out = torch.empty_like(in_4)
    _bn_add_kernel[(B, C)](
        in_4, in_0, in_1, in_3, in_2, in_5, relu_out,
        C, H * W, 1e-5,
    )
    return relu_out