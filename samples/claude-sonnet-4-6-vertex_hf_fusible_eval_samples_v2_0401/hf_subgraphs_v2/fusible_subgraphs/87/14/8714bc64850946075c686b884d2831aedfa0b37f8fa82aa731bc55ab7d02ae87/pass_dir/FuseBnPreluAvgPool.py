import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuse inference batch_norm + per-channel prelu into one kernel.
# Only prelu_out is returned — the single observable output consumed both by
# the model's direct return and as input to adaptive_avg_pool2d.
# ---------------------------------------------------------------------------

def pattern(x, running_mean, running_var, weight, bias, prelu_weight):
    bn_out    = torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    prelu_out = torch.prelu(bn_out, prelu_weight)
    return prelu_out


def replacement_args(x, running_mean, running_var, weight, bias, prelu_weight):
    return (x, running_mean, running_var, weight, bias, prelu_weight)


# ---------------------------------------------------------------------------
# Triton kernel — 2-D grid (bc_idx, hw_block_idx).
#
# Key design decision: BLOCK_HW is capped at 1024.
#   • For HW=1024 (float16/4, float16/7): BLOCK_HW=1024 gives 1 exact tile,
#     100% efficiency; BLOCK_HW=2048 would give 50% waste and was the root
#     cause of autotune instability in earlier experiments.
#   • For HW=2304 (float32/5, float16/8, bfloat16/2): BLOCK_HW=1024 gives
#     3 tiles → more programs → higher SM parallelism than BLOCK_HW=2048
#     (2 tiles). Higher program count consistently wins on the A30.
#   • For HW=784 (float16/3): smaller BLOCK_HW options cover HW well.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=16, num_stages=3),
    ],
    key=['HW', 'BC'],
)
@triton.jit
def _bn_prelu_kernel(
    x_ptr, mean_ptr, var_ptr, bn_w_ptr, bn_b_ptr, prelu_w_ptr,
    out_ptr,
    C, HW, BC,
    BLOCK_HW: tl.constexpr,
):
    bc_idx       = tl.program_id(0)
    hw_block_idx = tl.program_id(1)
    c = bc_idx % C

    mean    = tl.load(mean_ptr    + c).to(tl.float32)
    var     = tl.load(var_ptr     + c).to(tl.float32)
    bn_w    = tl.load(bn_w_ptr    + c).to(tl.float32)
    bn_b    = tl.load(bn_b_ptr    + c).to(tl.float32)
    prelu_w = tl.load(prelu_w_ptr + c).to(tl.float32)

    inv_std = bn_w * tl.rsqrt(var + 1e-3)
    shift   = bn_b - mean * inv_std

    offs = hw_block_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW

    base  = bc_idx * HW
    x     = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y_bn  = x * inv_std + shift
    y_out = tl.where(y_bn > 0.0, y_bn, prelu_w * y_bn)
    tl.store(out_ptr + base + offs, y_out, mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper — @torch.fx.wrap makes it opaque to torch.compile/FX tracing.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def bn_prelu_fused(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    B, C, H, W = x.shape
    HW = H * W
    BC = B * C

    out  = torch.empty_like(x)
    grid = lambda meta: (BC, triton.cdiv(HW, meta['BLOCK_HW']))

    _bn_prelu_kernel[grid](
        x, running_mean, running_var, bn_weight, bn_bias, prelu_weight,
        out,
        C, HW, BC,
    )

    return out


def replacement_func():
    return bn_prelu_fused