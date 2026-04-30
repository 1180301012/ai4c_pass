import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: inference-mode batch_norm with fixed hyper-parameters
# ---------------------------------------------------------------------------

def pattern(x, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 1e-05
    )


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


# ---------------------------------------------------------------------------
# Triton kernel: channel-wise inference batch-norm
#
# Grid: 2-D (N*C, ceil(HW / BLOCK_HW))
#   dim-0: one program per (batch, channel) pair  → loads 4 scalar params
#   dim-1: tiled over the spatial H*W elements
#
# Autotune keyed on (C, HW) so the same (C, HW) pair always reuses the
# best cached BLOCK_HW regardless of batch size.
# Autotune selects: BLOCK_HW=1024 for large HW, BLOCK_HW=256 for small HW.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_HW": 512},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_HW": 256},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_HW": 128},  num_warps=4, num_stages=2),
    ],
    key=["C", "HW"],
)
@triton.jit
def _bn_inf_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)   # index into N*C dimension
    pid_hw = tl.program_id(1)   # tile index along H*W

    # Channel index — fast for power-of-2 C (bitwise AND)
    c = pid_nc % C

    # Per-channel scalars (fp32 for numerical accuracy)
    mean  = tl.load(mean_ptr   + c).to(tl.float32)
    var   = tl.load(var_ptr    + c).to(tl.float32)
    w     = tl.load(weight_ptr + c).to(tl.float32)
    b     = tl.load(bias_ptr   + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    scale   = inv_std * w
    shift   = b - mean * scale

    # Spatial tile
    hw_off  = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask    = hw_off < HW
    base    = pid_nc * HW

    x = tl.load(x_ptr + base + hw_off, mask=mask).to(tl.float32)
    y = x * scale + shift

    tl.store(out_ptr + base + hw_off, y.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_bn_inference(x, running_mean, running_var, weight, bias):
    N, C, H, W = x.shape
    HW = H * W

    out = torch.empty_like(x)

    # 2-D grid: dim-0 = N*C programs (one per (n,c) pair)
    #           dim-1 = ceil(HW / BLOCK_HW) tiles
    grid = lambda meta: (N * C, triton.cdiv(HW, meta["BLOCK_HW"]))

    _bn_inf_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        C, HW,
    )

    return out


# ---------------------------------------------------------------------------
# Required entry-point
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_bn_inference