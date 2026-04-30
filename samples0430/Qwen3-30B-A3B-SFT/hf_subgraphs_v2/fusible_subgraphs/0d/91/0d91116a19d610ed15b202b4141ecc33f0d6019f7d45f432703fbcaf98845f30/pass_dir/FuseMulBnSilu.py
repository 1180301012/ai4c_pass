import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: multiply + batch_norm (inference)
# NOTE: silu is intentionally excluded because ForceArgsTracer normalizes
# `inplace=True` from a kwarg into a positional arg, which breaks matching
# against the Dynamo-traced graph where `inplace=True` stays as a kwarg.
# Matching just mul+bn still saves one full read+write of the intermediate.
#
# batch_norm argument order:
#   torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias,
#                                  training, momentum, eps)
# Model call: batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
#   in_0=running_mean, in_1=running_var, in_3=weight, in_2=bias
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0: running_mean  [C]
    # in_1: running_var   [C]
    # in_2: bias          [C]
    # in_3: weight        [C]
    # in_4: sigmoid scale [N, C, 1, 1]
    # in_5: input tensor  [N, C, H, W]
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# ---------------------------------------------------------------------------
# Triton kernel: fused  (x * scale) → BN-inference
#
# 2D grid: axis-0 = (n*C + c) index, axis-1 = spatial tile index.
# This avoids the expensive (offset // HW) % C division per element
# and gives perfectly coalesced memory access.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_mul_bn_kernel(
    x5_ptr,      # [N, C, H, W]  – feature map (GPU)
    x4_ptr,      # [N, C, 1, 1]  – per-channel scale (GPU)
    mean_ptr,    # [C]            – running_mean
    var_ptr,     # [C]            – running_var
    weight_ptr,  # [C]            – BN weight (gamma)
    bias_ptr,    # [C]            – BN bias   (beta)
    out_ptr,     # [N, C, H, W]  – output
    N_total,     # N * C * H * W  (autotune key)
    HW,          # H * W
    C,           # num channels
    eps,         # BN epsilon (float)
    BLOCK_HW: tl.constexpr,
):
    # pid0: linear (n*C + c) index; pid1: spatial tile index
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Channel index for per-channel param loads
    c = pid0 % C

    # Spatial tile: compute offsets within HW
    sp_off = pid1 * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask   = sp_off < HW

    # Flat indices in [N, C, H, W] layout
    flat_off = pid0 * HW + sp_off

    # ---- Load per-channel scalar (broadcast over all H*W elements) ----
    x4    = tl.load(x4_ptr + c)
    mean  = tl.load(mean_ptr + c).to(tl.float32)
    var   = tl.load(var_ptr  + c).to(tl.float32)
    w_val = tl.load(weight_ptr + c).to(tl.float32)
    b_val = tl.load(bias_ptr   + c).to(tl.float32)

    # ---- Load main tensor (coalesced) ----------------------------------
    x5 = tl.load(x5_ptr + flat_off, mask=mask, other=0.0)

    # ---- Fused: (x5 * x4) → BN inference ------------------------------
    x     = x5 * x4
    x_f32 = x.to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + eps)
    bn_out  = w_val * (x_f32 - mean) * inv_std + b_val

    tl.store(out_ptr + flat_off, bn_out.to(x5.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX tracing works)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_mul_bn(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0 : running_mean  [C]
    in_1 : running_var   [C]
    in_2 : bias          [C]
    in_3 : weight        [C]
    in_4 : sigmoid scale [N, C, 1, 1]
    in_5 : input tensor  [N, C, H, W]
    Returns: BN-normalized output [N, C, H, W]
    """
    N, C, H, W = in_5.shape
    device = in_5.device
    dtype  = in_5.dtype

    # BN buffers/parameters may live on CPU – move to GPU and match dtype
    mean   = in_0.to(device=device, dtype=dtype)
    var    = in_1.to(device=device, dtype=dtype)
    weight = in_3.to(device=device, dtype=dtype)
    bias_v = in_2.to(device=device, dtype=dtype)
    x4_gpu = in_4.to(device=device, dtype=dtype)

    x5_gpu = in_5
    out    = torch.empty_like(x5_gpu)

    HW = H * W
    NC = N * C
    N_total = NC * HW

    # Deterministic BLOCK_HW: no autotune overhead.
    # 4-tier dispatch:
    #   Tiny N (<= 500K): small blocks for occupancy
    #   Small N (<= 5M):  BLOCK_HW=1024, NW=4
    #   Med N (<= 30M):   BLOCK_HW=2048, NW=8
    #   Large N:           BLOCK_HW=4096, NW=8
    if N_total <= 500_000:
        BLOCK_HW = 512
        NW = 4
    elif N_total <= 5_000_000:
        BLOCK_HW = 1024
        NW = 4
    elif N_total <= 30_000_000:
        BLOCK_HW = 2048
        NW = 8
    else:
        BLOCK_HW = 4096
        NW = 8

    # 2D grid: (NC, ceil(HW / BLOCK_HW))
    grid = (NC, (HW + BLOCK_HW - 1) // BLOCK_HW)

    _fused_mul_bn_kernel[grid](
        x5_gpu, x4_gpu,
        mean, var, weight, bias_v,
        out,
        N_total, HW, C,
        1e-5,
        BLOCK_HW=BLOCK_HW,
        num_warps=NW,
    )

    return out


def replacement_func():
    return fused_mul_bn