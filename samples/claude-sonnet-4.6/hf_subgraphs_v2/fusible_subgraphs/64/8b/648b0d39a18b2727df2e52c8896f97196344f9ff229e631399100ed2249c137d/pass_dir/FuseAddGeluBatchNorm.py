import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: batch_norm(inference) -> 0+x  (single observable output: tmp_7)
# tmp_5 is the gelu output - it enters as an INPUT to this pattern
# tmp_6 is used ONLY inside the pattern (by 0+x) - NOT an external output
# tmp_7 is the only external output (used by model return)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, tmp_5):
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, tmp_5):
    return (in_0, in_1, in_2, in_3, tmp_5)


# ---------------------------------------------------------------------------
# Triton batch-norm kernel (inference mode) using FMA form
# out = x * scale + shift
#   where scale = weight / sqrt(var + eps),  shift = bias - mean * scale
# Grid: (N*C, ceil(HW / BLOCK_HW)) — one per (n,c) spatial tile.
# Per-channel params loaded as scalars (no gather).  All math in float32.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=4),
    ],
    key=['HW', 'C'],
)
@triton.jit
def _triton_bn_kernel(
    in_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    c        = pid_nc % C
    nc_off   = pid_nc * HW
    hw_start = pid_hw * BLOCK_HW

    hw_range = hw_start + tl.arange(0, BLOCK_HW)
    offsets  = nc_off + hw_range
    mask     = hw_range < HW

    x     = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Load per-channel params once per tile (scalar broadcast)
    mean = tl.load(mean_ptr   + c).to(tl.float32)
    var  = tl.load(var_ptr    + c).to(tl.float32)
    w    = tl.load(weight_ptr + c).to(tl.float32)
    b    = tl.load(bias_ptr   + c).to(tl.float32)

    # FMA form: scale = w/sqrt(var+eps), shift = b - mean*scale
    inv_std = 1.0 / tl.math.sqrt(var + 1e-5)
    scale   = w * inv_std
    shift   = b - mean * scale

    out = (x_f32 * scale + shift).to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper — @torch.fx.wrap so FX treats it as a leaf during tracing.
# in_0 : running_mean [C]   in_1 : running_var [C]
# in_2 : bias [C]           in_3 : weight [C]
# tmp_5: input tensor [N, C, H, W]  (gelu output)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_bn_identity(in_0, in_1, in_2, in_3, tmp_5):
    N  = tmp_5.shape[0]
    C  = tmp_5.shape[1]
    H  = tmp_5.shape[2]
    W  = tmp_5.shape[3]
    HW = H * W
    NC = N * C

    out = torch.empty_like(tmp_5)

    grid = lambda meta: (NC, triton.cdiv(HW, meta['BLOCK_HW']))

    _triton_bn_kernel[grid](
        tmp_5,
        in_0, in_1, in_3, in_2,   # mean, var, weight, bias
        out,
        C, HW,
    )

    return out


def replacement_func():
    return triton_bn_identity