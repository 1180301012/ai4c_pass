import torch
import triton
import triton.language as tl


# Match only batch_norm (inference).
# `x` is a placeholder wildcard — it will match the mul_result node in the model.
# We do NOT match silu here to avoid the ForceArgsTracer kwargs-normalization bug
# (F.silu(x, inplace=True) becomes args=(x,True) after normalization but model has
#  args=(x,), kwargs={inplace:True}, causing a length mismatch in SubgraphMatcher).
# The model's silu node remains and is applied AFTER our replacement BN output.
def pattern(x, in_0, in_1, in_2, in_3):
    tmp_5 = torch.nn.functional.batch_norm(x, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5


def replacement_args(x, in_0, in_1, in_2, in_3):
    return (x, in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def _fused_bn_kernel(
    x_ptr,      # [N, C, H, W]  – input (already multiplied in_5 * in_4)
    mean_ptr,   # [C]           – BN running_mean  (float32 on GPU)
    var_ptr,    # [C]           – BN running_var   (float32 on GPU)
    weight_ptr, # [C]           – BN weight        (float32 on GPU)
    bias_ptr,   # [C]           – BN bias          (float32 on GPU)
    out_ptr,    # [N, C, H, W]  – output (BN result; silu applied by model)
    N, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 2-D grid: axis-0 = N * ceil_div(HW, BLOCK_SIZE),  axis-1 = C
    pid = tl.program_id(0)
    c   = tl.program_id(1)

    n_spatial_blocks = tl.cdiv(HW, BLOCK_SIZE)
    n             = pid // n_spatial_blocks
    spatial_block = pid %  n_spatial_blocks

    # ── Load per-channel BN parameters (float32) ─────────────────────
    mean   = tl.load(mean_ptr   + c).to(tl.float32)
    var    = tl.load(var_ptr    + c).to(tl.float32)
    w      = tl.load(weight_ptr + c).to(tl.float32)
    b      = tl.load(bias_ptr   + c).to(tl.float32)

    # ── Spatial tile ─────────────────────────────────────────────────
    spatial_start = spatial_block * BLOCK_SIZE
    offsets       = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask          = offsets < HW

    base   = n * C * HW + c * HW
    x_vals = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # ── BN inference (float32 for accuracy) ──────────────────────────
    x_f32    = x_vals.to(tl.float32)
    inv_std  = 1.0 / tl.sqrt(var + eps)
    bn_out   = (x_f32 - mean) * inv_std * w + b

    # ── Store (cast back to original dtype) ──────────────────────────
    tl.store(out_ptr + base + offsets,
             bn_out.to(x_vals.dtype),
             mask=mask)


@torch.fx.wrap
def fused_bn(x, in_0, in_1, in_2, in_3):
    """
    Args (matching replacement_args order):
        x    : input tensor  [N, C, H, W]  (CUDA, = in_5 * in_4 already computed)
        in_0 : running_mean  [C]           (CPU)
        in_1 : running_var   [C]           (CPU)
        in_2 : bias          [C]           (CPU)
        in_3 : weight        [C]           (CPU)
    Returns: batch-normalized tensor (silu will be applied by the model's existing node)
    """
    device = x.device
    N, C, H, W = x.shape
    HW = H * W

    mean   = in_0.to(device=device, dtype=torch.float32)
    var    = in_1.to(device=device, dtype=torch.float32)
    weight = in_3.to(device=device, dtype=torch.float32)
    bias   = in_2.to(device=device, dtype=torch.float32)

    out = torch.empty_like(x)

    grid = lambda meta: (N * triton.cdiv(HW, meta['BLOCK_SIZE']), C)

    _fused_bn_kernel[grid](
        x,
        mean, var, weight, bias,
        out,
        N, C, HW,
        1e-5,
    )

    return out


def replacement_func():
    return fused_bn