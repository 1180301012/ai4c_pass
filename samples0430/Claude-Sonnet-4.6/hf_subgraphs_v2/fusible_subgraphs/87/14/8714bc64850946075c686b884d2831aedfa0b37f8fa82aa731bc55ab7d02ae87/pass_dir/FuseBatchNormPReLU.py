import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['HW'],
    warmup=5,
    rep=20,
)
@triton.jit
def _bn_prelu_fused_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    prelu_weight_ptr,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: axis-0 → (n, c) combined index, axis-1 → spatial tile
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Derive channel index from the (n, c) combined index
    c = pid_nc % C

    # Load per-channel BN parameters, cast to float32 for numerical stability
    mean   = tl.load(running_mean_ptr  + c).to(tl.float32)
    var    = tl.load(running_var_ptr   + c).to(tl.float32)
    bn_w   = tl.load(bn_weight_ptr     + c).to(tl.float32)
    bn_b   = tl.load(bn_bias_ptr       + c).to(tl.float32)
    prelu_w = tl.load(prelu_weight_ptr + c).to(tl.float32)

    # Spatial tile offsets
    hw_offs = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = hw_offs < HW

    # Base pointer for this (n, c) slice in NCHW layout
    base = pid_nc * HW

    # Load input (may be float16/bfloat16/float32)
    x     = tl.load(input_ptr + base + hw_offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # BN inference: y = (x - mean) * rsqrt(var + eps) * bn_w + bn_b
    x_norm = (x_f32 - mean) * tl.rsqrt(var + 1e-3)
    y      = x_norm * bn_w + bn_b

    # PReLU: out = y  if y >= 0
    #              y * prelu_w  otherwise
    out = tl.where(y >= 0.0, y, y * prelu_w)

    # Cast back to original input dtype and store
    tl.store(output_ptr + base + hw_offs, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def bn_prelu_fused(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    """Fused BN (inference) + PReLU kernel."""
    N, C, H, W = x.shape
    HW = H * W
    output = torch.empty_like(x)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    _bn_prelu_fused_kernel[grid](
        x,
        output,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        prelu_weight,
        C,
        HW,
    )

    return output


# ---------------------------------------------------------------------------
# Pattern to match:  batch_norm(x, ...) followed by prelu(bn_out, prelu_w)
# Must mirror the exact call signatures in model.py.
# ---------------------------------------------------------------------------
def pattern(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    bn_out   = torch.nn.functional.batch_norm(
        x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 0.001
    )
    prelu_out = torch.prelu(bn_out, prelu_weight)
    return prelu_out


def replacement_args(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (x, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


def replacement_func():
    return bn_prelu_fused