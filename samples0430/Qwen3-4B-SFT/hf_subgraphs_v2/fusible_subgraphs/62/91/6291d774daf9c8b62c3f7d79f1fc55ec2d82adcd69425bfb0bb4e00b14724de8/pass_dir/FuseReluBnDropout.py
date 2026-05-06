import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: ReLU -> BatchNorm (inference) -> Dropout (no-op p=0.0)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    in_0: running_mean  [C]
    in_1: running_var   [C]
    in_2: bias          [C]
    in_3: weight        [C]
    in_4: input tensor  [N, C]
    """
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ---------------------------------------------------------------------------
# Triton kernel: fused ReLU + BatchNorm (inference) in a single pass
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['C', 'N'],
)
@triton.jit
def _fused_relu_bn_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row (channel-group).
    Loads per-channel BN params once, applies ReLU then BN in a single pass.

    x          : [N, C]  — input features
    mean       : [C]     — running_mean
    var        : [C]     — running_var
    weight     : [C]     — BN weight (gamma)
    bias       : [C]     — BN bias  (beta)
    out        : [N, C]  — output
    eps        : scalar  — batch-norm epsilon  (1e-5)
    """
    row = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_SIZE)
    mask   = offs_c < C

    # ---- per-channel BN parameters (loaded once per program) -------------
    mean  = tl.load(mean_ptr  + offs_c, mask=mask, other=0.0)
    var   = tl.load(var_ptr   + offs_c, mask=mask, other=1.0)
    w     = tl.load(weight_ptr + offs_c, mask=mask, other=1.0)
    b     = tl.load(bias_ptr  + offs_c, mask=mask, other=0.0)

    # BN in inference: y = (x - mean) / sqrt(var + eps) * weight + bias
    # Precompute affine coefficients in fp32 for numerical stability
    inv_std = tl.rsqrt(var + 1e-5)
    factor  = w * inv_std                    # scale  per channel
    shift   = b - mean * factor             # bias adjustment

    # ---- per-element data ------------------------------------------------
    base    = row * C
    x       = tl.load(x_ptr + base + offs_c, mask=mask, other=0.0)

    # ReLU
    x = tl.maximum(x, 0.0)

    # BatchNorm (fp32 accumulation then cast back to input dtype)
    out     = x * factor + shift

    # Store (Triton auto-converts fp32 → output tensor dtype when storing)
    tl.store(out_ptr + base + offs_c, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_relu_bn(in_0, in_1, in_2, in_3, in_4):
    """
    Fused ReLU + BatchNorm (inference) + Dropout(p=0).

    Args:
        in_0: running_mean   [C]
        in_1: running_var    [C]
        in_2: bias           [C]
        in_3: weight         [C]
        in_4: input tensor   [N, C]  on CUDA

    Returns:
        out: [N, C]  on same device as in_4
    """
    N, C = in_4.shape
    eps  = 1e-5

    # Parameters live on CPU; move them to the same device as the activations
    dev = in_4.device
    x   = in_4.to(dev)
    mean = in_0.to(dev)
    var  = in_1.to(dev)
    w    = in_3.to(dev)   # in_3 = weight (gamma)
    b    = in_2.to(dev)   # in_2 = bias  (beta)

    out = torch.empty_like(x)

    grid = lambda meta: ((N * C + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_relu_bn_kernel[grid](
        x, mean, var, w, b, out,
        N, C,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement factory
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_relu_bn