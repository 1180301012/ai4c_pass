import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: inference BN (without learnable params) + residual add
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _bn_inf_residual_kernel(
    conv_out_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    residual_ptr,
    out_ptr,
    C,
    HW,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    For each element in the flat NCHW tensor:
      out = bn_inormalize(conv_out) + residual
    where bn_inormalize(x) = (x - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Channel index for each element
    c = (offsets // HW) % C

    # Load inputs (promote to fp32 for numerical stability)
    x = tl.load(conv_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Load per-channel BN parameters
    mean_val = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var_val  = tl.load(var_ptr  + c, mask=mask, other=1.0).to(tl.float32)
    gamma_v  = tl.load(gamma_ptr + c, mask=mask, other=1.0).to(tl.float32)
    beta_v   = tl.load(beta_ptr  + c, mask=mask, other=0.0).to(tl.float32)

    # BN inference: (x - mean) / sqrt(var + eps) * gamma + beta
    # Pre-compute scale and bias terms
    inv_std = tl.rsqrt(var_val + 1e-5)
    scale   = gamma_v * inv_std
    bias    = beta_v - mean_val * scale

    result = (x * scale + bias + r).to(tl.float16)

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_bn_add_bnfirst(conv_out, mean, var, gamma, beta, residual):
    """
    Computes: (conv_out - mean) / sqrt(var + eps) * gamma + beta + residual
    Argument order matches Pattern A: (conv_out, mean, var, gamma, beta, residual)
    """
    N, C, H, W = conv_out.shape
    HW = H * W
    n_elements = N * C * HW

    out = torch.empty_like(conv_out)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _bn_inf_residual_kernel[grid](
        conv_out, mean, var, gamma, beta, residual, out,
        C, HW, n_elements,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern: batch_norm output first, then add residual
# Matches deeppose graphs: `tmp_6 += in_6`  (tmp_6 = bn(conv), tmp_6 + in_6)
# ---------------------------------------------------------------------------
def pattern(conv_out, mean, var, gamma, beta, residual):
    bn_out = torch.nn.functional.batch_norm(conv_out, mean, var, gamma, beta, False, 0.1, 1e-05)
    result = bn_out + residual
    return (result,)


def replacement_args(conv_out, mean, var, gamma, beta, residual):
    return (conv_out, mean, var, gamma, beta, residual)


def replacement_func():
    return fused_bn_add_bnfirst